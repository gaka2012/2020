#include<stdio.h>
#include<stdlib.h>
#include<io.h>
#include "dirent.h"
#include "string.h"
#include "time.h"
#include "math.h"

/**
int main(int argc, char *argv[]) {

    int n;

    if (argc < 4) {
        printf("Usage: %s <SAC_file> <pick_file>\n", argv[0]);
        printf("  Picks are appended to end of <pick_file> in NLLOC_OBS format. \n");
        return 0;
    }
    double fw;
    fw=atof(argv[3]);
    printf("%f\n",fw);
    return 0;
}
**/
/**
int main()  
{     
    float n=0;
    int  i;
    char date[30]="20171231 2229   12.2307";
    char date1[30] = "20171231 2228   10.07";
    char min[10],sec[15],min1[10],sec1[15];
    int nmin,nmin1;
    float nsec,nsec1,dtime;
    
    //第一个时间值
    strncpy(min,date+11,2); //分钟
    strncpy(sec,date+16,7); //秒数
    //第二个时间值
    strncpy(min1,date1+11,2);
    strncpy(sec1,date1+16,7);

    printf("date == %s\n",date);
    
    nmin = atoi(min);
    nsec = atof(sec);
    nmin1 = atoi(min1);
    nsec1 = atof(sec1);
    //时间差
    dtime = abs(nmin1-nmin)*60+fabs(nsec1-nsec);
    printf("time difference is %f\n",dtime);    
    
    printf("test == %f\n",fabs(12-12.08));
    return 0;    
}  

**/

/**
int main()  
{     
    int n;
    FILE *fp1;
    char file_num[4][256]; //二维数组，第一个是数组的大小，第二个是每个组的内容大小。
    
    int  i;
    char *m_min;
    char m_sec[20];
    fp1 = fopen("test.txt","r");
    for (i=0;i<4;i++)
        fscanf(fp1,"%s",file_num[i]);
    fclose(fp1);
    for (i=0;i<4;i=i+2)
    {
        char *p_time;
        printf("%s\n",file_num[i]);
        strcpy(*p_time,file_num[i+1]);
        printf("%s\n",p_time);
        
        free(p_time);
        //strncpy(m_min,p_time+14,2);
        //strncpy(m_sec,p_time+17,11);
        //printf("%d %s\n",*m_min,m_sec);
    }   
    return 0;    
} 

**/



#define e 2.71828      



int main()  
{    
    float a=6.12,b=13.25;
    float test=0.466;
    test = (int)(100.0*test+0.5)/100.0;
    printf("test == %f\n",test);
    
    
    
    return 0;

}










