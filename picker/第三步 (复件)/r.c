#include<stdio.h>
#include<stdlib.h>
int main()
{
    //首先读取txt文件中的数据，存储到数组data中。
    
    float* data=calloc(15000000,sizeof(float));
    int num;
    int i=0;
    FILE *fp=fopen("zzp1.txt","r");
    if (!fp)
    {
        printf("error");
        return -1;
    }
    while(fscanf(fp,"%f",&data[i])!=EOF)
    {
        i++;
        //printf("%d\n",i);
    }
        
    fclose(fp);
    printf("%f\n",data[9999]);    
    //第二步，将数组中的数据写入到2进制文件中。
    FILE *bf;
    if ((bf=fopen("num2.txt","wb"))==NULL)
    {
        printf("can not open file");
        return -1;
    }
    fwrite(data,sizeof(float),i,bf);
    fclose(bf);
    
    //测试，读取2进制文件中的数据
    FILE *fp1;
    float* buffer=calloc(15000000,sizeof(float));
    //int j=10;
    if((fp1=fopen("num2.txt","rb"))==NULL)
    {
        printf("cant open the file");
        return -1;
    }
    fread(buffer,sizeof(float),i,fp1);   //可以一次读取
    printf("%f\n",buffer[9999]);
    //for(int m=0;m<i;m++)
    //    printf("%d\n",buffer[m]);   
    fclose(fp1); 
  
    return 0;
}
