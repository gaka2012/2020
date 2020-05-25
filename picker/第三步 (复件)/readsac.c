#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "sachdr.h"
#define PACKET_SIZE 10 //simulated packet size (seconds) //猜测的数据包的长度？？？？


/**用法：./readsac sacfile datafile 本程序将会读取sacfile中的头文件信息，并写入head.sac 同时会读取sacfile中的数据信息，利用datafile中的数值进行替换，最终写入head.sac . 因此，head.sac中有sacfile中的头文件信息以及datafile中的数据信息。
**/

int main(int argc, char *argv[]) {

    int n;

    if (argc < 3) {
        printf("Usage: %s <SAC_file> <pick_file>\n", argv[0]);
        printf("  Picks are appended to end of <pick_file> in NLLOC_OBS format. \n");
        return 0;
    }
    // open and read SAC file
    FILE *fp;
    if ((fp = fopen(argv[1], "r")) == 0) {
        perror(argv[1]);
        return -1;
    }
    // read header
    struct HDR sachdr;
    fread(&sachdr, sizeof (sachdr), 1, fp); //读取头文件
    fclose(fp);
    
    //将波形数据重新赋值
    FILE *fp1;
    int i;
    float* charFunctClipped=calloc(sachdr.NPTS,sizeof(float));
    //读入存放特征函数的文件
    if((fp1=fopen(argv[2],"rb"))==NULL) 
    {
        printf("cant open the file");
        return -1;
    }
    fread(charFunctClipped,sizeof(float),sachdr.NPTS,fp1);
    fclose(fp1);
    
            
    FILE *output=fopen("head.SAC","wb+");  //将读取的头文件写入sac文件中。
    fwrite(&sachdr,sizeof(struct HDR),1,output);
    fwrite(charFunctClipped,sizeof(float),sachdr.NPTS,output);
    fclose(output);
    //printf("num1==%f\n",wave[0]);
    
    free(charFunctClipped);
   
    
    
}





