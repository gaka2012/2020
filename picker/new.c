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
float cal_objective(int auto_pick,int m_pick,float d_list[],float t9)//计算目标函数，输入自动拾取的数量，手动拾取数量，时间差数组(与auto数量一致)，不确定性。
{
    float ob; //目标函数的值初始化为0
    float base_num; // 底数
    int i;
    if ( auto_pick<=3 & m_pick==1 & auto_pick>0) //参考文献中的第一中情况，惩罚参数是3,找到最小值min_diff
    {
        //1.找到最小值
        float min_diff=10;
        for(i=0;i<auto_pick;i++) if (d_list[i]<min_diff) min_diff=d_list[i];
        base_num = ((min_diff*min_diff)/(2*(t9*t9)))*(-1); 
        ob = pow (e,base_num); 
        //printf("newtest==%f\n",min_diff);
    }
    
    else if (auto_pick>3 & m_pick==1)//第二种情况
    {
        float tem,sum_tem; 
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


int main()  
{    
    int a =4,i;
    float b[a],test;
    for (i=0;i<a;i++) b[i]=(i+0.2)*0.2;
    for (i=0;i<a;i++) printf("%f\n",b[i]);
    //test=cal_objective(2,0,b,2);
    //printf("test == %f\n",test);
    char c[5]="-1234";
    if (strcmp(c,"-1234")==0)printf("char === %s\n",c);
    return 0;

}










