/*
 * This file is part of the Anthony Lomax C Library.
 *
 * Copyright (C) 2006-2012 Anthony Lomax <anthony@alomax.net www.alomax.net>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

/***************************************************************************
 * FilterPicker5.c:
 *
 * Anthony J Lomax
 *   ALomax Scientific www.alomax.net
 *
 * History/Changes:
 *
 *      ver 5.0.0       2008.07.14 AJL - originally based on FilterPicker5.java
 *      ver 5.0.1       2011.11.04 AJL - update released
 *      ver 5.0.2       2011.11.14 AJL - update released
 *      ver 5.0.3       2012.10.19 AJL - added PickData -> double polarityWeight, FilterPicker5_Memory -> double pickPolarityWeight
 *                                       and supporting code in this file to support a weight for the pick polarity
 *
 *
 ***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ew_bridge.h"
#include "PickData.h"
#include "FilterPicker5_Memory.h"
#include "FilterPicker5.h"

#define DEBUG 0
#define MAX_MSG_SIZE 1024
#define CRITICAL_POLARITY_RATIO 0.667

enum ResultType resultType = PICKS;


/*** function to calculate picks  */
// _DOC_ =============================
// _DOC_ FilterPicker5
// _DOC_ =============================

//public final float[] apply(double deltaTime, float[] sample) {

void Pick(

        double deltaTime, // dt or timestep of data samples
        float* sample, // array of num_samples data samples
        int num_samples, // the number of samples in array sample

        const double filterWindow, // FilterPicker5 filter window
        // _DOC_ the filter window (filterWindow) in seconds determines how far back in time the previous samples are examined.  The filter window will be adjusted upwards to be an integer N power of 2 times the sample interval (deltaTime).  Then numRecursive = N + 1 "filter bands" are created.  For each filter band n = 0,N  the data samples are processed through a simple recursive filter backwards from the current sample, and picking statistics and characteristic function are generated.  Picks are generated based on the maximum of the characteristic funciton values over all filter bands relative to the threshold values threshold1 and threshold2.
        const double longTermWindow, // FilterPicker5 long term window
        // _DOC_ the long term window (longTermWindow) determines: a) a stabilisation delay time after the beginning of data; before this delay time picks will not be generated. b) the decay constant of a simple recursive filter to accumlate/smooth all picking statistics and characteristic functions for all filter bands.
        const double threshold1, // FilterPicker5 threshold1
        // _DOC_ threshold1 sets the threshold to trigger a pick event (potential pick).  This threshold is reached when the (clipped) characteristic function for any filter band exceeds threshold1.
        const double threshold2, // FilterPicker5 threshold1
        // _DOC_ threshold2 sets the threshold to declare a pick (pick will be accepted when tUpEvent reached).  This threshold is reached when the integral of the (clipped) characteristic function for any filter band over the window tUpEvent exceeds threshold2 * tUpEvent (i.e. the average (clipped) characteristic function over tUpEvent is greater than threshold2)..
        const double tUpEvent, // FilterPicker5 tUpEvent
        // _DOC_ tUpEvent determines the maximum time the integral of the (clipped) characteristic function is accumlated after threshold1 is reached (pick event triggered) to check for this integral exceeding threshold2 * tUpEvent (pick declared).

        FilterPicker5_Memory** pmem, // memory structure/object
        // _DOC_ pointer to a memory structure/object is used so that this function can be called repetedly for packets of data in sequence from the same channel.
        // The calling routine is responsible for managing and associating the correct mem structures/objects with each channel.  On first call to this function for each channel set mem = NULL.
        BOOLEAN_INT useMemory, // set to TRUE_INT=1 if function is called for packets of data in sequence, FALSE_INT = 0 otherwise

        PickData*** ppick_list, // returned pointer to array of num_picks PickData structures/objects containing picks
        int* pnum_picks, // the number of picks in array *ppick_list
        char* channel_id // a string identifier for the data channel
        ) {

    int ishift, iPolarity;
    BOOLEAN_INT error1_not_printed = TRUE_INT;
    double charFunctValueTrigger = -1.0; // AJL 20091216
    int indexUpEventTrigger = -1;
    int indexUncertaintyPick = -1;
    int i, k, l, m, n;
    float* sampleNew = NULL;
    double integralCharFunctClippedWindow;
    double maxCharFunctValue;
    double charFunct = 0.0;
    double charFunctClipped = 0.0;
    double charFunctClippedTest = 0.0;
    double charFunctTest;
    double currentSample;
    double currentDiffSample;
    double currentFilteredSample;
    double currentDiffSample2;
    double polarityderivativeIncrement;
    double dev, dy;
    BOOLEAN_INT acceptedPick;
    BOOLEAN_INT upCharFunctUncertainty;
    int upEventBufPtrLast;
    int indexBeginPick;
    int indexEndPick;
    double triggerPeriod;
    double uncertainty;
    PickData* pickData;
    // 20121019 AJL
    double polDerivSum;
    double polSumAbsDeriv;
    double polDerivRatio;
    FILE *fp;

    // _DOC_ =============================
    // _DOC_ apply algoritm

    // initialize memory object
    FilterPicker5_Memory* mem = *pmem;
    if (mem == NULL) {
        mem = init_filterPicker5_Memory(deltaTime, sample, num_samples, filterWindow, longTermWindow, threshold1, threshold2, tUpEvent);
    }

    // create array for time-series results
    if (resultType == TRIGGER || resultType == CHAR_FUNC) {
        sampleNew = calloc(num_samples, sizeof (float));
        //sampleNew[0] = sample[num_samples - 1] = 0.0f;
    }

    // _DOC_ set clipped limit of maximum char funct value to 5 * threshold1 to avoid long recovery time after strong events
    maxCharFunctValue = 5.0 * threshold1;


    // _DOC_ =============================
    // _DOC_ loop over all samples
    for (n = 0; n < num_samples; n++) {  //n < num_samples
        acceptedPick = FALSE_INT; // 一开始pick是没有的。

        // _DOC_ update index of nTUpEvent length up event window buffers
        upEventBufPtrLast = mem->upEventBufPtr; //0,1,2,3,4......20
        mem->upEventBufPtr = (mem->upEventBufPtr + 1) % mem->nTUpEvent; //1,2,3,4....0
        // _DOC_ =============================
        // _DOC_ characteristic function is  (E2 - mean_E2) / mean_stdDev_E2
        // _DOC_    where E2 = (filtered band value current - filtered band value previous)**2
        // _DOC_    where value previous is taken futher back for longer filter bands
        charFunct = 0.0;
        charFunctClipped = 0.0;
        // _DOC_ evaluate current signal values
        currentSample = sample[n];
        // _DOC_ filters are applied to first difference of signal values
        currentDiffSample = currentSample - mem->lastSample;
        // _DOC_ loop over numRecursive filter bands
        for (k = mem->numRecursive - 1; k >= 0; k--) {
            // _DOC_  apply two single-pole HP filters
            // _DOC_  http://en.wikipedia.org/wiki/High-pass_filter    y[i] := α * (y[i-1] + x[i] - x[i-1])
            currentFilteredSample = mem->highPassConst[k] * (mem->filteredSample[k][0] + currentDiffSample);
            currentDiffSample2 = currentFilteredSample - mem->filteredSample[k][0];
            mem->filteredSample[k][0] = currentFilteredSample;
            currentFilteredSample = mem->highPassConst[k] * (mem->filteredSample[k][1] + currentDiffSample2);
            mem->filteredSample[k][1] = currentFilteredSample;
            // _DOC_  apply one single-pole LP filter
            // _DOC_  http://en.wikipedia.org/wiki/Low-pass_filter    y[i] := y[i-1] + α * (x[i] - y[i-1])
            currentFilteredSample = mem->filteredSample[k][2] + mem->lowPassConst[k] * (currentFilteredSample - mem->filteredSample[k][2]);
            mem->lastFilteredSample[k] = mem->filteredSample[k][2];
            mem->filteredSample[k][2] = currentFilteredSample;
            dy = currentFilteredSample;
            /* TEST */ //
            mem->test[k] = dy;
            //
            mem->xRec[k] = dy * dy;
            charFunctClippedTest = 0.0; // AJL 20091214
            if (mem->mean_stdDev_xRec[k] <= DOUBLE_MIN_VALUE) {
                if (mem->enableTriggering && error1_not_printed) {
                    sprintf(message_str, "WARNING: %s: mem->mean_stdDev_xRec[k] <= Float.MIN_VALUE (this should not happen! - dead trace?)\n", channel_id);
                    info(message_str);
                    error1_not_printed = FALSE_INT;
                }
            } else {  //205行这个else是必然要经过的过程。
                charFunctTest = (mem->xRec[k] - mem->mean_xRec[k]) / mem->mean_stdDev_xRec[k];
                charFunctClippedTest = charFunctTest; // AJL 20091214
                // _DOC_ limit maximum char funct value to avoid long recovery time after strong events
                if (charFunctClippedTest > maxCharFunctValue) { //避免clipped大于50
                    charFunctClippedTest = maxCharFunctValue;
                    // save corrected mem->xRec[k]
                    mem->xRec[k] = maxCharFunctValue * mem->mean_stdDev_xRec[k] + mem->mean_xRec[k];
                }
                // _DOC_ characteristic function is maximum over numRecursive filter bands
                if (charFunctTest >= charFunct) {  //charFunct第一个总是0，貌似后面有初始化。//第一个if
                    charFunct = charFunctTest;
                    charFunctClipped = charFunctClippedTest; //找到charFuncTest的最大值，并将charFunctClippedTest赋值给clipped，因为clipped的值是被限制过的值。后面的积分就是用的clipped
                    mem->charFunctNumRecursiveIndex[mem->upEventBufPtr] = k;
                    //printf("k1==%d\n",k);
                }
                // _DOC_ trigger index is highest frequency with CF >= threshold1 over numRecursive filter bands
                if (charFunctTest >= threshold1) {
                    mem->charFunctNumRecursiveIndex[mem->upEventBufPtr] = k;
                   // printf("k2==%d\n",k);
                }
            } /////////////////else结束///////////////
            // AJL 20091214
            // _DOC_ =============================
            // _DOC_ update uncertainty and polarity fields
            // _DOC_ uncertaintyThreshold is at minimum char function or char funct increases past uncertaintyThreshold
            mem->charFunctUncertainty[k] = charFunctClippedTest; // no smoothing
            // AJL 20091214 mem->charFunctLast = charFunctClipped;
            upCharFunctUncertainty =
                    ((mem->charFunctUncertaintyLast[k] < mem->uncertaintyThreshold[k]) && (mem->charFunctUncertainty[k] >= mem->uncertaintyThreshold[k]));
            mem->charFunctUncertaintyLast[k] = mem->charFunctUncertainty[k];
            // _DOC_ each time characteristic function rises past uncertaintyThreshold store sample index and initiate polarity algoirithm
            if (upCharFunctUncertainty) {
                mem->indexUncertainty[k][mem->upEventBufPtr] = n - 1;
            } else {
                mem->indexUncertainty[k][mem->upEventBufPtr] = mem->indexUncertainty[k][upEventBufPtrLast];
            }
            // END - AJL 20091214
            if (upCharFunctUncertainty) {
                // _DOC_ initialize polarity algorithm, uses derivative of signal
                mem->polarityDerivativeSum[k][mem->upEventBufPtr] = 0.0;
                mem->polaritySumAbsDerivative[k][mem->upEventBufPtr] = 0.0;
            } else {
                mem->polarityDerivativeSum[k][mem->upEventBufPtr] = mem->polarityDerivativeSum[k][upEventBufPtrLast];
                mem->polaritySumAbsDerivative[k][mem->upEventBufPtr] = mem->polaritySumAbsDerivative[k][upEventBufPtrLast];
            }
            // _DOC_   accumulate derivative and sum of abs of derivative for polarity estimate
            // _DOC_   accumulate since last indexUncertainty
            polarityderivativeIncrement = mem->filteredSample[k][2] - mem->lastFilteredSample[k];
            mem->polarityDerivativeSum[k][mem->upEventBufPtr] += polarityderivativeIncrement;
            mem->polaritySumAbsDerivative[k][mem->upEventBufPtr] += fabs(polarityderivativeIncrement);
        }   //numRecursive结束/////////////////
//////2 在这写入特征函数        
        //fp=fopen("zzp1.txt","a+");
/*        fprintf(fp,"n==%d   ",n);*/
        //fprintf(fp,"%f\n",charFunctClipped);
        //fclose(fp);

        // _DOC_ =============================
        // _DOC_ trigger and pick logic //从刚开始，过一段稳定时间之后才能pick
        // _DOC_ only apply trigger and pick logic if past stabilisation time (longTermWindow) 
        if (mem->enableTriggering || mem->nTotal++ > mem->indexEnableTriggering) { // past stabilisation time 
            //初始enableTriggering是false，indexEnableTriggering=1 + (int) (longTermWindow / deltaTime); nTotal=-1
            mem->enableTriggering = TRUE_INT;
          //  printf("这个if被激活了\n");

            // _DOC_ update charFunctClipped values, subtract oldest value, and save provisional current sample charFunct value
            // _DOC_ to avoid spikes, do not use full charFunct value, may be very large, instead use charFunctClipped
            mem->integralCharFunctClipped[mem->upEventBufPtr] =
                    mem->integralCharFunctClipped[upEventBufPtrLast] - mem->charFunctClippedValue[mem->upEventBufPtr] + charFunctClipped;
/*            fp=fopen("zzp.txt","a+");*/
/*            fprintf(fp,"1==%f ",mem->integralCharFunctClipped[mem->upEventBufPtr]);*/
/*            fprintf(fp,"2==%f ",mem->integralCharFunctClipped[upEventBufPtrLast]);*/
/*            fprintf(fp,"3==%f ",mem->charFunctClippedValue[mem->upEventBufPtr]);*/
/*            fprintf(fp,"4==%f\n",charFunctClipped);*/
/*            fclose(fp);*/
            
            mem->charFunctClippedValue[mem->upEventBufPtr] = charFunctClipped; 
            mem->charFunctValue[mem->upEventBufPtr] = charFunct;//这几步实现了移动窗口一步一步的移动。

            // _DOC_ if new picks allowd, check if integralCharFunct over last tUpEvent window is greater than threshold
            if (mem->allowNewPickIndex != INT_UNSET && mem->integralCharFunctClipped[mem->upEventBufPtr] >= mem->criticalIntegralCharFunct) { //                                                                 21*threshold2
                             //allowNewPickIndex等于一个正常的数值才允许进行pick
                // _DOC_ find last point in tUpEvent window where charFunct rose past threshold1 and integralCharFunct greater than threshold back to this point
                m = mem->upEventBufPtr;
              //  printf("mem->upEventBufPtr==%d\n",mem->upEventBufPtr);
                integralCharFunctClippedWindow = mem->charFunctClippedValue[m];
              // printf("mem->charFunctClippedValue[m]==%f\n",mem->charFunctClippedValue[m]);
                k = 0;
              //  printf("n==%d\n",n);
                while (k++ < mem->nTUpEvent - 1 && n - k > mem->allowNewPickIndex) {
                    m--;
                   // printf("m==%d ",m);  //m恒等于19个值。
                    if (m < 0) {
                        m += mem->nTUpEvent;
                    }
//3 显示叠加值
/*                    fp=fopen("zzp.txt","a+");*/
/*                    fprintf(fp,"n==%d ",n);*/
/*                    fprintf(fp,"mem->charFunctClippedValue[m]==%f\n",mem->charFunctClippedValue[m]);*/
/*                    //fprintf(fp,"mem->xRec[k]==%f\n",mem->xRec[k]);*/
                   // fclose(fp);
                    integralCharFunctClippedWindow += mem->charFunctClippedValue[m]; //叠加？？积分？？
                 //   printf("ntegralCharFunctClippedWindow==%f\n",integralCharFunctClippedWindow);
                    if (mem->charFunctValue[m] >= threshold1) {  //312行
                        l = m - 1;
                        if (l < 0) {
                            l += mem->nTUpEvent;
                        }
                        //printf("l==%d\n",l);
                        if (mem->charFunctValue[l] < threshold1) { //很远的一个if
                           // printf("这个if是成立的AND==%f\n",mem->charFunctValue[l]);
                            //printf("l==%d\n",l);
                            // integralCharFunct is integralCharFunct from current point back to point m
                            if (integralCharFunctClippedWindow >= mem->criticalIntegralCharFunct) { //21×threshold2
  //     1                      //  printf("mem->criticalIntegralCharFunct==%f\n",mem->criticalIntegralCharFunct);
                                acceptedPick = TRUE_INT;
                               // printf("这个if是成立的==%d %f\n",n,integralCharFunctClippedWindow);
                                // _DOC_ save characteristic function value as indicator of pick strenth
                                charFunctValueTrigger = mem->charFunctValue[m]; // AJL 20091216
                              //  printf("m==%d\n",m);
                                mem->triggerNumRecursiveIndex = mem->charFunctNumRecursiveIndex[m];
                                // _DOC_ set index for pick uncertainty begin and end
                                indexUpEventTrigger = n - k;
                             //   printf("k==%d\n",k);
                                indexUncertaintyPick = mem->indexUncertainty[mem->triggerNumRecursiveIndex][m]; // AJ
                              //  printf("mem->triggerNumRecursiveIndex==%d\n",mem->triggerNumRecursiveIndex);
                                // _DOC_ evaluate polarity based on accumulated derivative
                                // _DOC_    (=POS if derivative_sum > 0, = NEG if derivative_sum < 0,
                                // _DOC_     and if ratio larger abs derivative_sum / abs_derivative_sum > 0.667,
                                // _DOC_     =UNK otherwise)
                                iPolarity = m + 1; // evaluate polarity at 1 point past trigger point
                                if (iPolarity >= mem->nTUpEvent) {
                                    iPolarity -= mem->nTUpEvent;
                                }
                                // 20121019 AJL - following modified to add pickPolarityWeight
                                mem->pickPolarity = POLARITY_UNKNOWN;
                                mem->pickPolarityWeight = 0.0;                                      //m+1,triggerh后面一点
                                polDerivSum = mem->polarityDerivativeSum[mem->triggerNumRecursiveIndex][iPolarity];
                                polSumAbsDeriv = mem->polaritySumAbsDerivative[mem->triggerNumRecursiveIndex][iPolarity];
                                polDerivRatio = polDerivSum / polSumAbsDeriv;
                                if (polDerivSum > 0.0 && polDerivRatio > CRITICAL_POLARITY_RATIO) {
                                    mem->pickPolarity = POLARITY_POS;
                                    mem->pickPolarityWeight = (polDerivRatio - CRITICAL_POLARITY_RATIO) / (1.0 - CRITICAL_POLARITY_RATIO);
                                } else if (polDerivSum < 0.0 && -polDerivRatio > CRITICAL_POLARITY_RATIO) {
                                    mem->pickPolarity = POLARITY_NEG;
                                    mem->pickPolarityWeight = (-polDerivRatio - CRITICAL_POLARITY_RATIO) / (1.0 - CRITICAL_POLARITY_RATIO);
                                }
                                // END - 20121019 AJL
                                info(message_str);
                                mem->allowNewPickIndex = INT_UNSET;
                                info(message_str);
                                break;
                            }
                        }
                    }
                } //261行的while
            }

            // _DOC_ if no pick, check if charFunctUncertainty has dropped below threshold maxAllowNewPickThreshold to allow new picks
            //printf("mem->allowNewPickIndex==%d\n",mem->allowNewPickIndex);
/*            fp=fopen("zzp.txt","a+");*/
/*            fprintf(fp,"mem->allowNewPickIndex==%d\n",mem->allowNewPickIndex);*/
/*            fprintf(fp,"n==%d\n",n);*/
/*           // fprintf(fp,"mem->xRec[k]==%f\n",mem->xRec[k]);*/
/*            fclose(fp);*/
            if (!acceptedPick && mem->allowNewPickIndex == INT_UNSET) { // no pick and no allow new picks
                // AJL 20091214                             //一个乱七八糟的值
                
/*                fp=fopen("zzp.txt","a+");*/
/*                fprintf(fp,"这个if是成立的==%d\n",n);*/
/*                //fprintf(fp,"mem->xRec[k]==%f\n",mem->xRec[k]);*/
/*                fclose(fp);*/
                k = 0;
                for (; k < mem->numRecursive; k++) {
                    if (mem->charFunctUncertainty[k] > mem->maxAllowNewPickThreshold) // do not allow new picks
                    {                                  //2
                        break;
                    }
                }    //上面这个for如果都没有经历break这个阶段，那么最终k就会等于10,或者说如果10个滤波后的特征值都满足小于
                     //2的话，就允许继续pick
                if (k == mem->numRecursive) {
                    mem->allowNewPickIndex = n;
                }
                // END AJL 20091214
            }
        }    //if (mem->enableTriggering || mem->nTotal++ > mem->indexEnableTriggeri


        // _DOC_ =============================
        // _DOC_ update "true", long-term statistic based on current signal values based on long-term window
        // long-term decay formulation                   variance:方差
        // _DOC_ update long-term means of x, dxdt, E2, var(E2), uncertaintyThreshold
        for (k = 0; k < mem->numRecursive; k++) {
          //  fp=fopen("zzp.txt","a+");
        //    fprintf(fp,"mem->mean_xRec[k]==%f for循环\n",mem->mean_xRec[k]);
          
        //longDecayFactor = deltaTime / longTermWindow; longDecayConst = 1.0 - filterPicker5_Memory->longDecayFactor;
            mem->mean_xRec[k] = mem->mean_xRec[k] * mem->longDecayConst + mem->xRec[k] * mem->longDecayFactor;
                          //0.998                          //0.02
         //   fprintf(fp,"mem->mean_xRec[k]==%f 第二个\n",mem->mean_xRec[k]);
           // fclose(fp);  
            dev = mem->xRec[k] - mem->mean_xRec[k];  //当前值减去均值。
            mem->mean_var_xRec[k] = mem->mean_var_xRec[k] * mem->longDecayConst + dev * dev * mem->longDecayFactor;
            // _DOC_ mean_stdDev_E2 is sqrt(long-term mean var(E2))
            //stddev==standard deviation:标准差,离散度，背景噪声的离散度相当，有信号时，短时间窗口离散度扩大。
            mem->mean_stdDev_xRec[k] = sqrt(mem->mean_var_xRec[k]);
            mem->uncertaintyThreshold[k] = mem->uncertaintyThreshold[k] * mem->longDecayConst + mem->charFunctUncertainty[k] * mem->longDecayFactor;
            if (mem->uncertaintyThreshold[k] > mem->maxUncertaintyThreshold) {
                mem->uncertaintyThreshold[k] = mem->maxUncertaintyThreshold;
            } else if (mem->uncertaintyThreshold[k] < mem->minUncertaintyThreshold) {
                mem->uncertaintyThreshold[k] = mem->minUncertaintyThreshold;
            }
        }


        // _DOC_ =============================
        //  _DOC_ act on result, save pick if pick accepted at this sample

        if (resultType == TRIGGER) { // show triggers
          //  printf("resultType==%d\n",resultType);
            if (acceptedPick) {
                sampleNew[n] = 1.0f;
            } else {
                sampleNew[n] = 0.0f;
            }
            // TEST...
            //sampleNew[n] = (float) mem->test[0];
            //sampleNew[n] = (float) mem->test[index_recursive];
            //sampleNew[n] = (float) mem->test[mem->numRecursive - 1];
            //
        } else if (resultType == CHAR_FUNC) { // show char function
           // printf("elseif==%d\n",CHAR_FUNC);
            sampleNew[n] = (float) charFunctClipped;
        } else { // generate picks
            // PICK
            if (acceptedPick) {
                // _DOC_ if pick accepted, save pick time, uncertainty, strength (integralCharFunct) and polarity
                // _DOC_    pick time is uncertainty threshold (characteristic function rose past
                // _DOC_    uncertaintyThreshold) and trigger time (characteristic function >= threshold1)
                // _DOC_    pick begin is pick time - (trigger time - uncertainty threshold)
                indexBeginPick = indexUncertaintyPick - (indexUpEventTrigger - indexUncertaintyPick);
                indexEndPick = indexUpEventTrigger;
                triggerPeriod = mem->period[mem->triggerNumRecursiveIndex];
                // check that uncertainty range is >= triggerPeriod / 20.0  // 20101014 AJL
                uncertainty = deltaTime * ((double) (indexEndPick - indexBeginPick));
                if (uncertainty < triggerPeriod / 20.0) {
                    ishift = (int) (0.5 * (triggerPeriod / 20.0 - uncertainty) / deltaTime);
                    // advance uncertainty index
                    indexBeginPick -= ishift;
                    // delay trigger index
                    indexEndPick += ishift;
                }
              
                pickData = init_PickData();
                set_PickData(pickData, (double) indexBeginPick, (double) indexEndPick,
                        mem->pickPolarity, mem->pickPolarityWeight, charFunctValueTrigger, // AJL 20091216
                        CHAR_FUNCT_AMP_UNITS, triggerPeriod);
                addPickToPickList(pickData, ppick_list, pnum_picks);

            }
        }


        mem->lastSample = currentSample;
        mem->lastDiffSample = currentDiffSample;

    }


    if (useMemory) {
        // corect memory index values for sample length
        for (i = 0; i < mem->nTUpEvent; i++) {
            // AJL 20091214
            for (k = 0; k < mem->numRecursive; k++) {
                mem->indexUncertainty[k][i] -= num_samples;
            }
            // END - AJL 20091214
        }
        if (mem->allowNewPickIndex != INT_UNSET) {
            mem->allowNewPickIndex -= num_samples;
        }
    } else {
        free_FilterPicker5_Memory(&mem);
        mem = NULL;
    }
    *pmem = mem;


    if (resultType == TRIGGER || resultType == CHAR_FUNC) {
        for (n = 0; n < num_samples; n++)
            sample[n] = sampleNew[n];
    }
    if (sampleNew != NULL)
        free(sampleNew);


}







