#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#include "sachdr.h"

#include "ew_bridge.h"
#include "PickData.h"
#include "FilterPicker5_Memory.h"
#include "FilterPicker5.h"

#define DEBUG 0
//clear all;
//close all;

void MonthDay(int year, int yearday, int* pmonth, int* pday);

int main(int argc, char *argv[]) {

    //改1 原本是argc<3
    if (argc < 8) {
        printf("Usage: %s <SAC_file> <pick_file>\n", argv[0]);
        printf("  Picks are appended to end of <pick_file> in NLLOC_OBS format. \n");
        return 0;
    }



    BOOLEAN_INT useMemory = FALSE_INT; // set to TRUE_INT (=1) if function is called for packets of data in sequence, FALSE_INT (=0) otherwise
     
    //改2共12行.以下是修改的参数，改成手动输入,顺序为filterwindow，longtermwindow，tupevent，t1,t2
    double fw;
    double lt;
    double threshold1;
    double threshold2;
    double tUpEvent;
    double t0;
    double filterWindow;
    double longTermWindow;
    char file_num[4][256]; //二维数组，第一个是数组的大小，第二个是每个组的内容大小。用来读取txt文件中的内容，每行需要2个数组，一个存放路径，一个存放拾取到时。
    int  i;
    FILE *fp1;
    
    fw=atof(argv[3]);
    lt=atof(argv[4]);
    t0=atof(argv[5]);
    threshold1=atof(argv[6]);
    threshold2=atof(argv[7]);
    //test1=atof(argv[3]);
    //printf("test: %f\n",test1);
    // open and read SAC file
    FILE *fp;
    //打开txt文件，将txt中的内容输出到字符串数组中。txt每行内容有2部分，占用2个数组。
    fp1 = fopen(argv[1],"r"); 
    for (i=0;i<4;i++)
        fscanf(fp1,"%s",file_num[i]);
    fclose(fp1);
    
    //对读入到数组中的内容进行循环。
    for (i=0;i<4;i=i+2)
    {
        printf("%s\n",file_num[i]);
    
    //读取sac文件的绝对路径。
        if ((fp = fopen(file_num[i], "r")) == 0) {
            perror(argv[1]);
            printf ("can not open sac files\n");
            return -1;
        }
    // read header
        struct HDR sachdr;
        fread(&sachdr, sizeof (sachdr), 1, fp);
    // allocate array for data
        if (DEBUG) printf("sachdr.NPTS: %d\n", sachdr.NPTS);
        float* sample = calloc(sachdr.NPTS, sizeof (float));
    // read data
        fread(sample, sizeof (float), sachdr.NPTS, fp);
        fclose(fp);



    // set picker paramters (TODO: make program arguments?)
    // SEE: _DOC_ in FilterPicker5.c for more details on the picker parameters
    // defaults
    // filp_test filtw 4.0 ltw 10.0 thres1 8.0 thres2 8.0 tupevt 0.2 res PICKS...
    //改3。 下面5行被注释了
    //double filterWindow = 4.0; // NOTE: auto set below
    //double longTermWindow = 10.0; // NOTE: auto set below
    //double threshold1 = 12.0;
    //double threshold2 = 10.0;
    //double tUpEvent = 0.5; // NOTE: auto set below
    //
    // auto set values
    // get dt
        double dt = sachdr.DELTA;
        if (DEBUG) printf("sachdr.DELTA: %f\n", sachdr.DELTA);
    //dt = dt < 0.02 ? 0.02 : dt;     // aviod too-small values for high sample rate data
    //改4. 原本的被注释了，重新改成了第2行。
    //filterWindow = 300.0 * dt; //2**8=256 2**9=512 2**7=128
        filterWindow=fw*dt;
        long iFilterWindow = (long) (0.5 + filterWindow * 1000.0);
        if (iFilterWindow > 1)
            filterWindow = (double) iFilterWindow / 1000.0;
    //改5。 原本的被注释了，重新改成了第2行。
    //longTermWindow = 300.0 * dt; // seconds
        longTermWindow = lt * dt;
        long ilongTermWindow = (long) (0.5 + longTermWindow * 1000.0);
        if (ilongTermWindow > 1)
            longTermWindow = (double) ilongTermWindow / 1000.0;
    //改6. 原本的被注释了，重新改成了第2行。
    //tUpEvent = 31.0 * dt; // time window to take integral of charFunct version
        tUpEvent = t0 * dt;
        long itUpEvent = (long) (0.5 + tUpEvent * 1000.0);
        if (itUpEvent > 1)
            tUpEvent = (double) itUpEvent / 1000.0;
    //
        printf("picker_func_test: filp_test filtw %f ltw %f thres1 %f thres2 %f tupevt %f res PICKS\n",
                filterWindow, longTermWindow, threshold1, threshold2, tUpEvent);



    // do picker function test
        PickData** pick_list = NULL; // array of num_picks ptrs to PickData structures/objects containing returned picks
        int num_picks = 0;
        FilterPicker5_Memory* mem = NULL;

        Pick(
                sachdr.DELTA,
                sample,
                sachdr.NPTS,
                filterWindow,
                longTermWindow,
                threshold1,
                threshold2,
                tUpEvent,
                &mem,
                useMemory,
                &pick_list,
                &num_picks,
                "TEST"
                );
        printf("picker_func_test: num_picks: %d\n", num_picks);

    // create NLLOC_OBS picks
    // open pick file
    
        if ((fp = fopen(argv[2], "a")) == 0) {
            perror(argv[2]);
            return -1;
        }
    // date
        int month, day;
        MonthDay(sachdr.NZYEAR, sachdr.NZJDAY, &month, &day);
        double sec = (double) sachdr.B + (double) sachdr.NZSEC + (double) sachdr.NZMSEC / 1000.0;
    // id fields
        char onset[] = "?";
        char* kstnm;
        kstnm = calloc(1, 16 * sizeof (char));
        strncpy(kstnm, sachdr.KSTNM, 6);
        char* kinst;
        kinst = calloc(1, 16 * sizeof (char));
        strncpy(kinst, sachdr.KINST, 6);
        if (strstr(kinst, "(count") != NULL)
            strcpy(kinst, "(counts)");
        char* kcmpnm;
        kcmpnm = calloc(1, 16 * sizeof (char));
        strncpy(kcmpnm, sachdr.KCMPNM, 6);
        char phase[16];
    // create NLL picks
        char* pick_str;
        pick_str = calloc(1, 1024 * sizeof (char));
        int n;
        
        //自动拾取的到时的分钟和秒数。
        char pick_sec[20],m_min[4];

        //手动拾取的到时以及分钟和秒数，并转换为整数或小数。
        char p_time[50],m_sec[20],pick_min[4];
        int ma_min,p_min;
        float ma_sec,p_sec,dtime;

        //存放时间差的数组。
        float d_list[num_picks];

        for (n = 0; n < num_picks; n++) {
            
            
            sprintf(phase, "P%d_", n);
            pick_str = printNlloc(pick_str,
                    *(pick_list + n), sachdr.DELTA, kstnm, kinst, kcmpnm, onset, phase,
                    sachdr.NZYEAR, month, day, sachdr.NZHOUR, sachdr.NZMIN, sec);
        // write pick to <pick_file> in NLLOC_OBS format
            
            //strncpy(pick_date,pick_str+32,23); //20171231 2229   12.2307
            //strncpy(pick_un,pick_str+60,9);    //3.000e-02 不确定性
            
            //pick_st = "AXI    -12345 BHZ    ? P0_    ? 20171231 2229  -47.0293 GAU 3.900e-01 0.000e+00 5.274e+00 2.560e+00"
            //自动拾取的分钟和秒数。
            strncpy(pick_min,pick_str+43,2);   
            strncpy(pick_sec,pick_str+47,8);  
            p_min = atoi(pick_min);
            p_sec = atof(pick_sec);
            
            //txt文件中的手动拾取的到时以及其分钟和秒数。
            strcpy(p_time,file_num[i+1]);
            strncpy(m_min,p_time+14,2);
            strncpy(m_sec,p_time+17,11);
            ma_min = atoi(m_min);
            ma_sec = atof(m_sec);
            
            //计算时间差(只计算分钟和秒数)(自动减去手动)
            dtime = fabs((p_min-ma_min)*60+(p_sec-ma_sec));
            d_list[n] = dtime;
            fprintf(fp, "%s\n", pick_str);
            
        }
        for (n = 0; n < num_picks; n++) printf("%f\n",d_list[n]);
        // clean up
        fclose(fp);
        //free(pick_min);
        free(pick_str);
        free(kcmpnm);
        free(kinst);
        free(kstnm);
        free_FilterPicker5_Memory(&mem);
        free_PickList(pick_list, num_picks);
        free(sample);
    }
    return (0);

}



/** date functions */

static char daytab[2][13] = {
    {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

/** function to set month / day from day of year */

void MonthDay(int year, int yearday, int* pmonth, int* pday) {
    int i, leap;

    leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    for (i = 1; yearday > daytab[leap][i]; i++)
        yearday -= daytab[leap][i];
    *pmonth = i;
    *pday = yearday;

}

