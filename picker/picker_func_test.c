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
#define e 2.71828
//clear all;
//close all;

void MonthDay(int year, int yearday, int* pmonth, int* pday);
float cal_objective(int auto_pick,int m_pick,float d_list[],float t9);

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
    char file_num[34][256]; //二维数组，第一个是数组的大小，第二个是每个组的内容大小。用来读取txt文件中的内容，每行需要2个数组，一个存放路径，一个存放拾取到时。
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
    float sum_ob=0,tem_ob=0;  //定义目标函数的和,以及每一个trace的目标函数。
    
    //打开txt文件，将txt中的内容输出到字符串数组中。txt每行内容有2部分，占用2个数组。
    fp1 = fopen(argv[1],"r"); 
    for (i=0;i<34;i++)
        fscanf(fp1,"%s",file_num[i]);
    fclose(fp1);
    
    //对读入到数组中的内容进行循环。
    for (i=0;i<34;i=i+2)
    {
        printf("%s\n",file_num[i]); //print读入的文件名
    
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
        float *tt = sachdr.T;
        printf("t9==%f\n",tt[9]);
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
            if (strcmp(file_num[i+1],"-1234")!=0) //如果路径后面不是-1234,则说明是地震，则计算时间差以及目标函数。
            {
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
            }
            fprintf(fp, "%s\n", pick_str);
            
        }
        //计算目标函数。
        if (strcmp(file_num[i+1],"-1234")!=0) //如果后缀不是-1234,表明是地震，则目标函数的第二个输入值是1,代表地震。
        {
            //for (n = 0; n < num_picks; n++) printf("%f\n",d_list[n]);
            tem_ob = cal_objective(num_picks,1,d_list,tt[9]);
            printf("tem_ob==%f\n",tem_ob);
        }
        else if (strcmp(file_num[i+1],"-1234")==0) //如果后缀是-1234,表明是噪声，则目标函数的第二个输入值是0,即Nm==0表示噪声。  
        {   
            tem_ob = cal_objective(num_picks,0,d_list,tt[9]);
            printf("noise_tem_ob==%f\n",tem_ob);
        }
        sum_ob+=tem_ob; //计算目标函数的和。
        
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
    printf("sum_ob==%f\n",sum_ob);
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
float cal_objective(int auto_pick,int m_pick,float d_list[],float t9)//计算目标函数，输入自动拾取的数量，手动拾取数量，时间差数组(与auto数量一致)，不确定性。
{
    float ob=0; //目标函数的值初始化为0
    float base_num=0; // 底数
    int i;
    if ( auto_pick<=3 & m_pick==1 & auto_pick>0) //参考文献中的第一中情况，惩罚参数是3,找到最小值min_diff
    {
        
        //1.找到最小值
        float min_diff=10;
        for(i=0;i<auto_pick;i++) if (d_list[i]<min_diff) min_diff=d_list[i];
        base_num = ((min_diff*min_diff)/(2*(t9*t9)))*(-1); 
        printf("base_num==%f %f\n",base_num,min_diff);
        ob = pow (e,base_num); 
        //printf("newtest==%f\n",min_diff);
    }
    
    else if (auto_pick>3 & m_pick==1)//第二种情况
    {
        float tem=0,sum_tem=0; 
        for (i=0;i<auto_pick;i++)
        {
            base_num = (d_list[i]*d_list[i]/(2*(t9*t9)))*(-1);
            tem      = pow(e,base_num);
            sum_tem+= tem; 
        }
        ob = sum_tem/(1+auto_pick-1);
    }
    else if (auto_pick==0 & m_pick==1) ob=0;
    else if (m_pick==0) 
    {
        ob=pow(1.0/(auto_pick+4),auto_pick+1); //第四种情况。
    }
    return ob;
}

