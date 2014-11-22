/*
 * motiondetect.c
 *
 *  Copyright (C) Georg Martius - February 1007-2011
 *   georg dot martius at web dot de
 *  Copyright (C) Alexey Osipov - Jule 2011
 *   simba at lerlan dot ru
 *   speed optimizations (threshold, spiral, SSE, asm)
 *
 *  This file is part of vid.stab video stabilization library
 *
 *  vid.stab is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License,
 *  as published by the Free Software Foundation; either version 2, or
 *  (at your option) any later version.
 *
 *  vid.stab is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with GNU Make; see the file COPYING.  If not, write to
 *  the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 */
#include "motiondetect.h"
#include "motiondetect_internal.h"
#include "motiondetect_opt.h"
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "boxblur.h"
#include "vidstabdefines.h"
#include "localmotion2transform.h"
#include "transformtype_operations.h"
#include "transformtype_operations.h"

#define USE_SPIRAL_FIELD_CALC

int contrast = 400;
int contrastLookup[256];
int contrastLookupInitialized = 0;
int derivativeKernelSize = 4;
float derivativeKernel[4][4];
int derivativeKernelInitialized = 0;

/* internal data structures */

// structure that contains the contrast and the index of a field
typedef struct _contrast_idx {
  double contrast;
  int index;
} ContrastContainer;


VSMotionDetectConfig vsMotionDetectGetDefaultConfig(const char* modName){
  VSMotionDetectConfig conf;
  conf.stepSize          = 6;
  conf.accuracy          = 15;
  conf.shakiness         = 5;
  conf.virtualTripod     = 0;
  conf.contrastThreshold = 0.25;
  conf.show              = 0;
  conf.modName           = modName;
  return conf;
}

void vsMotionDetectGetConfig(VSMotionDetectConfig* conf, const VSMotionDetect* md){
  if(md && conf)
    *conf = md->conf;
}

const VSFrameInfo* vsMotionDetectGetFrameInfo(const VSMotionDetect* md){
  return &md->fi;
}


int vsMotionDetectInit(VSMotionDetect* md, const VSMotionDetectConfig* conf, const VSFrameInfo* fi){
  assert(md && fi);
  md->conf = *conf;
  md->fi = *fi;

  if(fi->pFormat<=PF_NONE ||  fi->pFormat==PF_PACKED || fi->pFormat>=PF_NUMBER) {
    vs_log_warn(md->conf.modName, "unsupported Pixel Format (%i)\n",
                md->fi.pFormat);
    return VS_ERROR;
  }

  vsFrameAllocate(&md->prev, &md->fi);
  if (vsFrameIsNull(&md->prev)) {
    vs_log_error(md->conf.modName, "malloc failed");
    return VS_ERROR;
  }

  vsFrameNull(&md->curr);
  vsFrameNull(&md->currorig);
  vsFrameNull(&md->currtmp);
  md->hasSeenOneFrame = 0;
  md->frameNum = 0;

  // TODO: get rid of shakiness parameter in the long run
  md->conf.shakiness = VS_MIN(10,VS_MAX(1,md->conf.shakiness));
  md->conf.accuracy = VS_MIN(15,VS_MAX(1,md->conf.accuracy));
  if (md->conf.accuracy < md->conf.shakiness / 2) {
    vs_log_info(md->conf.modName, "Accuracy should not be lower than shakiness/2 -- fixed");
    md->conf.accuracy = md->conf.shakiness / 2;
  }
  if (md->conf.accuracy > 9 && md->conf.stepSize > 6) {
    vs_log_info(md->conf.modName, "For high accuracy use lower stepsize  -- set to 6 now");
    md->conf.stepSize = 6; // maybe 4
  }

  int minDimension = VS_MIN(md->fi.width, md->fi.height);
//  shift: shakiness 1: height/40; 10: height/4
//  md->maxShift = VS_MAX(4,(minDimension*md->conf.shakiness)/40);
//  size: shakiness 1: height/40; 10: height/6 (clipped)
//  md->fieldSize = VS_MAX(4,VS_MIN(minDimension/6, (minDimension*md->conf.shakiness)/40));

  // fixed size and shift now
  int maxShift      = VS_MAX(16, minDimension/7);
  int fieldSize     = VS_MAX(16, minDimension/10);
  int fieldSizeFine = VS_MAX(6, minDimension/60);
#if defined(USE_SSE2) || defined(USE_SSE2_ASM)
  fieldSize     = (fieldSize / 16 + 1) * 16;
  fieldSizeFine = (fieldSizeFine / 16 + 1) * 16;
#endif
  if (!initFields(md, &md->fieldscoarse, fieldSize, maxShift, md->conf.stepSize,
                  1, 0, md->conf.contrastThreshold)) {
    return VS_ERROR;
  }
  // for the fine check we use a smaller size and smaller maximal shift (=size)
  if (!initFields(md, &md->fieldsfine, fieldSizeFine, fieldSizeFine,
                  2, 1, fieldSizeFine, md->conf.contrastThreshold/2)) {
    return VS_ERROR;
  }

  vsFrameAllocate(&md->curr,&md->fi);
  vsFrameAllocate(&md->currtmp, &md->fi);

  md->initialized = 2;
  return VS_OK;
}

void vsMotionDetectionCleanup(VSMotionDetect* md) {
  if(md->fieldscoarse.fields) {
    vs_free(md->fieldscoarse.fields);
    md->fieldscoarse.fields=0;
  }
  if(md->fieldsfine.fields) {
    vs_free(md->fieldsfine.fields);
    md->fieldsfine.fields=0;
  }
  vsFrameFree(&md->prev);
  vsFrameFree(&md->curr);
  vsFrameFree(&md->currtmp);

  md->initialized = 0;
}

// returns true if match of local motion is better than threshold
short lm_match_better(void* thresh, void* lm){
  if(((LocalMotion*)lm)->match <= *((double*)thresh))
    return 1;
  else
    return 0;
}

// TODO [DEBUGGING] Detect motion
int vsMotionDetection(VSMotionDetect* md, LocalMotions* motions, VSFrame *frame) {
// fprintf(stderr, "Processing frame %i\n", md->frameNum);
 assert(md->initialized==2);

	md->currorig = *frame;
	// smoothen image to do better motion detection
	//  (larger stepsize or eventually gradient descent (need higher resolution))
	if (md->fi.pFormat > PF_PACKED) {
		// we could calculate a grayscale version and use the PLANAR stuff afterwards
		// so far smoothing is only implemented for PLANAR
		vsFrameCopy(&md->curr, frame, &md->fi);
	} else {
		//=-- Copy frame into curr
		vsFrameCopy(&md->curr, frame, &md->fi);

//		boxblurPlanar(&md->curr, &md->curr, &md->currtmp, &md->fi, md->conf.stepSize * 2, BoxBlurNoColor);

		//=-- Use derivative as exposure correction mechanism
		calculateDerivativeImage(md->curr.data[0], md->curr.data[0], &md->currtmp, &md->fi, md->curr.linesize[0], md->curr.linesize[0]);

		//=-- Smooth result for better matching
		boxblurPlanar(&md->curr, &md->curr, &md->currtmp, &md->fi, md->conf.stepSize * 2, BoxBlurNoColor);
//		boxblurPlanar(&md->curr, &md->curr, &md->currtmp, &md->fi, md->conf.stepSize * 2, BoxBlurNoColor);

	}

  //=-- TODO [CUSTOM&TEMP] See preprocessed frame in motion vector output
	if (md->conf.show) {
		vsFrameCopy(&md->currorig, &md->curr, &md->fi);
	}

	if (md->hasSeenOneFrame) {
		LocalMotions motionscoarse;
		LocalMotions motionsfine;
		vs_vector_init(&motionsfine, 0);
		//    md->curr = frame;
		if (md->fi.pFormat > PF_PACKED) {
			motionscoarse = calcTransFields(md, &md->fieldscoarse,
					calcFieldTransPacked, contrastSubImgPacked);
		} else { // PLANAR
			motionscoarse = calcTransFields(md, &md->fieldscoarse,
					calcFieldTransPlanar, contrastSubImgPlanar);
		}
		int num_motions = vs_vector_size(&motionscoarse);

		if (num_motions < 1) {
			vs_log_warn(md->conf.modName,
					"too low contrast. \
(no translations are detected in frame %i)\n",
					md->frameNum);
		} else {
			// calc transformation and perform another scan with small fields
			VSTransform t = vsSimpleMotionsToTransform(md->fi, md->conf.modName,
					&motionscoarse);
			md->fieldsfine.offset = t;
			md->fieldsfine.useOffset = 1;
			LocalMotions motions2;
			if (md->fi.pFormat > PF_PACKED) {
				motions2 = calcTransFields(md, &md->fieldsfine,
						calcFieldTransPacked, contrastSubImgPacked);
			} else { // PLANAR
				motions2 = calcTransFields(md, &md->fieldsfine,
						calcFieldTransPlanar, contrastSubImgPlanar);
			}

			motionsfine = motions2;


			// throw out those with bad match (worse than mean of coarse scan)
//			if (0) {
//				printf("\nMatches: mean:  %f | ", meanMatch);
//				vs_array_print(matchQualities1, stdout);
//				printf("\n         fine: ");
//				VSArray matchQualities2 = localmotionsGetMatch(&motions2);
//				vs_array_print(matchQualities2, stdout);
//				printf("\n");
//			}

			//=-- TODO [CUSTOM] Remove motions that stray too much from the mean motion
			// TODO [High] This approach could fail for rotation

			//=- Calculate combined motions average and standard deviation
				VSArray matchQualities1 = localmotionsGetMatch(&motionscoarse);
				double meanMatch = cleanmean(matchQualities1.dat,
						matchQualities1.len, NULL, NULL);
				motionsfine = vs_vector_filter(&motions2, lm_match_better,
						&meanMatch);

			LocalMotionsInfo motionsAvgAndDeviation;
			if (num_motions < 10) {
				LocalMotions combinedMotions = (LocalMotions) vs_vector_concat(&motionscoarse, &motionsfine);
				motionsAvgAndDeviation = calculateMotionsAvgAndDeviation(&combinedMotions);
			} else {

				motionsAvgAndDeviation = calculateMotionsAvgAndDeviation(&motionscoarse);
			}
//
//			//=- Cleanup coarse and fine motions
			motionscoarse = cleanupMotions(&motionscoarse, &motionsAvgAndDeviation);
			motionsfine = cleanupMotions(&motionsfine, &motionsAvgAndDeviation);

		}

		if (md->conf.show) { // draw fields and transforms into frame.
			num_motions = vs_vector_size(&motionscoarse);
			int num_motions_fine = vs_vector_size(&motionsfine);
			// this has to be done one after another to handle possible overlap
			if (md->conf.show > 1) {
				for (int i = 0; i < num_motions; i++)
					drawFieldScanArea(md, LMGet(&motionscoarse, i),
							md->fieldscoarse.maxShift);
			}
			for (int i = 0; i < num_motions; i++) {
				LocalMotion* currentMotion = LMGet(&motionscoarse, i);
				drawField(md, currentMotion, 1);
			}
			for (int i = 0; i < num_motions_fine; i++) {
				LocalMotion* currentMotion = LMGet(&motionsfine, i);
				drawField(md, currentMotion, 0);
			}
			for (int i = 0; i < num_motions; i++) {
				LocalMotion* currentMotion = LMGet(&motionscoarse, i);
				drawFieldTrans(md, currentMotion, 180);
			}
			for (int i = 0; i < num_motions_fine; i++) {
				LocalMotion* currentMotion = LMGet(&motionsfine, i);
				drawFieldTrans(md, currentMotion, 64);
			}
		}

		*motions = vs_vector_concat(&motionscoarse, &motionsfine);

		//*motions = motionscoarse;
		//*motions = motionsfine;
	} else {
		vs_vector_init(motions, 1); // dummy vector
		md->hasSeenOneFrame = 1;
	}

  // for tripod we keep a certain reference frame
  if(md->conf.virtualTripod < 1 || md->frameNum < md->conf.virtualTripod)
    // copy current frame (smoothed) to prev for next frame comparison
    vsFrameCopy(&md->prev, &md->curr, &md->fi);
  md->frameNum++;
  return VS_OK;
}

// TODO [CUSTOM] Cleanup motions based on average vector
LocalMotionsInfo calculateMotionsAvgAndDeviation(LocalMotions* averageSource) {
//	VSArray averageSourceQualities = localmotionsGetMatch(&averageSource);
	int averageSourceMotionCount = vs_vector_size(averageSource);

	LocalMotionsInfo avgAndStdDev;

	//=- Calculate average motion
	double averageMotionX = 0;
	double averageMotionY = 0;
	double totalQualityWeight = 0;

//	fprintf(stderr,"Calculating average over %i motions\n", averageSourceMotionCount);
	for (int i = 0; i < averageSourceMotionCount; i++) {
		LocalMotion* currentMotion = LMGet(averageSource, i);
		Vec currentVector = currentMotion->vector;

		//=- Include quality as measure of how much the current motion should contribute to the average
		double currentQuality = currentMotion->match;
		double currentWeight = 1 / (currentQuality + 0.0001);
		totalQualityWeight += currentWeight;

//		fprintf(stderr,"Current vector [%i,%i] Q:%f \n", currentVector.x, currentVector.y, currentQuality);

		averageMotionX += currentVector.x * currentWeight;
		averageMotionY += currentVector.y * currentWeight;
	}
	averageMotionX = averageMotionX / totalQualityWeight;
	averageMotionY = averageMotionY / totalQualityWeight;

	avgAndStdDev.averageTranslationX = averageMotionX;
	avgAndStdDev.averageTranslationY = averageMotionY;

//	fprintf(stderr,"Average motion [%f,%f]\n", averageMotionX, averageMotionY);

	//=- Calculate standard deviation
	double standardDeviationX = 0;
	double standardDeviationY = 0;
	for (int i = 0; i < averageSourceMotionCount; i++) {
		LocalMotion* currentMotion = LMGet(averageSource, i);
		Vec currentVector = currentMotion->vector;

		standardDeviationX += pow((double)currentVector.x - averageMotionX, 2);
		standardDeviationY += pow((double)currentVector.y - averageMotionY, 2);
	}

	standardDeviationX = sqrt(standardDeviationX/averageSourceMotionCount);
	standardDeviationY = sqrt(standardDeviationY/averageSourceMotionCount);
//	fprintf(stderr,"Standard deviation [%f,%f]\n", standardDeviationX, standardDeviationY);

	avgAndStdDev.stdDevX = standardDeviationX;
	avgAndStdDev.stdDevY = standardDeviationY;

	return avgAndStdDev;
}


// TODO [CUSTOM] Cleanup motions based on average vector
LocalMotions cleanupMotions(LocalMotions* motions, LocalMotionsInfo* motionsInfo) {
	int motionCount = vs_vector_size(motions);

	//=- Actually remove straying motions
	LocalMotions cleanedMotions;
	vs_vector_init(&cleanedMotions,motionCount);

	for (int i = 0; i < motionCount; i++) {
		LocalMotion* currentMotion = LMGet(motions, i);
		Vec currentVector = currentMotion->vector;

		if (   abs((double)currentVector.x - motionsInfo->averageTranslationX) > motionsInfo->stdDevX/2
			|| abs((double)currentVector.y - motionsInfo->averageTranslationY) > motionsInfo->stdDevY/2) {
			//= Strays too much
//			fprintf(stderr,"Throwing away vector [%i,%i]\n", currentVector.x, currentVector.y);
		} else {
//			fprintf(stderr,"Adding vector [%i,%i]\n", currentVector.x, currentVector.y);
			vs_vector_append(&cleanedMotions, currentMotion);
		}
	}

	return cleanedMotions;
}

/** initialise measurement fields on the frame.
    The size of the fields and the maxshift is used to
    calculate an optimal distribution in the frame.
    if border is set then they are placed savely away from the border for maxShift
*/

int initFields(VSMotionDetect* md, VSMotionDetectFields* fs,
               int size, int maxShift, int stepSize,
               short keepBorder, int spacing, double contrastThreshold) {
  fs->fieldSize = size;
  fs->maxShift  = maxShift;
  fs->stepSize  = stepSize;
  fs->useOffset = 0;
  fs->contrastThreshold = contrastThreshold;

  int rows = VS_MAX(3,(md->fi.height - fs->maxShift*2)/(size+spacing)-1);
  int cols = VS_MAX(3,(md->fi.width - fs->maxShift*2)/(size+spacing)-1);
  // make sure that the remaining rows have the same length
  fs->fieldNum = rows * cols;
  fs->fieldRows = rows;

  if (!(fs->fields = (Field*) vs_malloc(sizeof(Field) * fs->fieldNum))) {
    vs_log_error(md->conf.modName, "malloc failed!\n");
    return 0;
  } else {
    int i, j;
    int border=fs->stepSize;
    // the border is the amount by which the field centers
    // have to be away from the image boundary
    // (stepsize is added in case shift is increased through stepsize)
    if(keepBorder)
      border = size / 2 + fs->maxShift + fs->stepSize;
    int step_x = (md->fi.width  - 2 * border) / VS_MAX(cols-1,1);
    int step_y = (md->fi.height - 2 * border) / VS_MAX(rows-1,1);
    for (j = 0; j < rows; j++) {
      for (i = 0; i < cols; i++) {
        int idx = j * cols + i;
        fs->fields[idx].x = border + i * step_x;
        fs->fields[idx].y = border + j * step_y;
        fs->fields[idx].size = size;
      }
    }
  }
  fs->maxFields = (md->conf.accuracy) * fs->fieldNum / 15;
  vs_log_info(md->conf.modName, "Fieldsize: %i, Maximal translation: %i pixel\n",
              fs->fieldSize, fs->maxShift);
  vs_log_info(md->conf.modName, "Number of used measurement fields: %i out of %i\n",
              fs->maxFields, fs->fieldNum);

  return 1;
}

/** \see contrastSubImg*/
double contrastSubImgPlanar(VSMotionDetect* md, const Field* field) {
#ifdef USE_SSE2
  return contrastSubImg1_SSE(md->curr.data[0], field, md->curr.linesize[0],md->fi.height);
#else
  return contrastSubImg(md->curr.data[0],field,md->curr.linesize[0],md->fi.height,1);
#endif

}

/**
   \see contrastSubImg_Michelson three times called with bytesPerPixel=3
   for all channels
*/
double contrastSubImgPacked(VSMotionDetect* md, const Field* field) {
  unsigned char* const I = md->curr.data[0];
  int linesize2 = md->curr.linesize[0]/3; // linesize in pixels
  return (contrastSubImg(I, field, linesize2, md->fi.height, 3)
          + contrastSubImg(I + 1, field, linesize2, md->fi.height, 3)
          + contrastSubImg(I + 2, field, linesize2, md->fi.height, 3)) / 3;
}

/**
 * TODO [Medium] Is the Michelson contrast the best choice, here? Why?
 *
   calculates Michelson-contrast in the given small part of the given image
   to be more compatible with the absolute difference formula this is scaled by 0.1

   \param I pointer to framebuffer
   \param field Field specifies position(center) and size of subimage
   \param width width of frame (linesize in pixels)
   \param height height of frame
   \param bytesPerPixel calc contrast for only for first channel
*/
double contrastSubImg(unsigned char* const I, const Field* field, int width,
                      int height, int bytesPerPixel) {
  int k, j;
  unsigned char* p = NULL;
  int s2 = field->size / 2;
  unsigned char mini = 255;
  unsigned char maxi = 0;

  p = I + ((field->x - s2) + (field->y - s2) * width) * bytesPerPixel;
  for (j = 0; j < field->size; j++) {
    for (k = 0; k < field->size; k++) {
      mini = (mini < *p) ? mini : *p;
      maxi = (maxi > *p) ? maxi : *p;
      p += bytesPerPixel;
    }
    p += (width - field->size) * bytesPerPixel;
  }
  return (maxi - mini) / (maxi + mini + 0.1); // +0.1 to avoid division by 0
}

/* calculates the optimal transformation for one field in Planar frames
 * (only luminance)
 */
// TODO [DEBUGGING] Do subimage matching
LocalMotion calcFieldTransPlanar(VSMotionDetect* md, VSMotionDetectFields* fs,
                                 const Field* field, int fieldnum) {
  int tx = 0;
  int ty = 0;
  uint8_t *Y_c = md->curr.data[0], *Y_p = md->prev.data[0];
  int linesize_c = md->curr.linesize[0], linesize_p = md->prev.linesize[0];
  // we only use the luminance part of the image
  int i, j;
  int stepSize = fs->stepSize;
  int maxShift = fs->maxShift;
  Vec offset = { 0, 0};
  LocalMotion lm = null_localmotion();

  if(fs->useOffset){
    // Todo: we could put the preparedtransform into fs
    PreparedTransform pt = prepare_transform(&fs->offset, &md->fi);
    Vec fieldpos = {field->x, field->y};
    offset = sub_vec(transform_vec(&pt, &fieldpos), fieldpos);
    // is the field still in the frame
    int s2 = field->size/2;
    if(unlikely(fieldpos.x+offset.x-s2-maxShift-stepSize < 0 ||
                fieldpos.x+offset.x+s2+maxShift+stepSize >= md->fi.width ||
                fieldpos.y+offset.y-s2-maxShift-stepSize < 0 ||
                fieldpos.y+offset.y+s2+maxShift+stepSize >= md->fi.height)){
      lm.match=-1;
      return lm;
    }
  }

#ifdef STABVERBOSE
  // printf("%i %i %f\n", md->frameNum, fieldnum, contr);
  FILE *fieldLM = NULL;
  char buffer[32];
  vs_snprintf(buffer, sizeof(buffer), "f%04i_%02i.dat", md->frameNum, fieldnum);
  fieldLM = fopen(buffer, "w");
  fprintf(fieldLM, "# splot \"%s\"\n", buffer);
#endif

#ifdef USE_SPIRAL_FIELD_CALC
  unsigned int minerror = UINT_MAX;

  const int MIN_ERROR_COUNT = 5;
  const unsigned int NO_ERROR = -1;
  unsigned int previousMinError[MIN_ERROR_COUNT];
  int previousMinErrorTX[MIN_ERROR_COUNT];
  int previousMinErrorTY[MIN_ERROR_COUNT];
  for (i = 0; i < MIN_ERROR_COUNT; i++) {
	  previousMinError[i] = NO_ERROR;
	  previousMinErrorTX[i] = 0;
	  previousMinErrorTY[i] = 0;
  }
  int previousMinErrorIndex = 0;

  // check all positions by outgoing spiral
  i = 0; j = 0;
  int limit = 1;
  int step = 0;
  int dir = 0;
  while (j >= -maxShift && j <= maxShift && i >= -maxShift && i <= maxShift) {
	  // TODO [CLEANUP]
	//unsigned char* const I1, unsigned char* const I2,
	// const Field* field,
	// int width1 = linesize_c,
	// int width2 = linesize_p,
	// int height = fi_height,
	// int bytesPerPixel = 1,
	// int d_x = i + offset.x,
	// int d_y = j +offset.y,
	// unsigned int threshold = minerror

    unsigned int error = compareSubImg(Y_c, Y_p, field, linesize_c, linesize_p,
                                       md->fi.height, 1, i + offset.x, j + offset.y,
                                       minerror);

       //= TODO [CLEANUP]
//	  if (md->frameNum == 5) {
//		  fprintf(stderr, "{%i}-[%i,%i]-(%i)\n", fieldnum, i, j, error);
//	  }

    if (error < minerror) {
      previousMinError[previousMinErrorIndex] = minerror;
	  previousMinErrorTX[previousMinErrorIndex] = i;
	  previousMinErrorTY[previousMinErrorIndex] = j;
      previousMinErrorIndex = (previousMinErrorIndex + 1)%MIN_ERROR_COUNT;
      minerror = error;
      tx = i;
      ty = j;
    }

    //=-- If the error is really low, we can stop early
    if (minerror < 20) {
    	break;
    }

    //spiral indexing...
    step++;
    switch (dir) {
     case 0:
      i += stepSize;
      if (step == limit) {
        dir = 1;
        step = 0;
      }
      break;
     case 1:
      j += stepSize;
      if (step == limit) {
        dir = 2;
        step = 0;
        limit++;
      }
      break;
     case 2:
      i -= stepSize;
      if (step == limit) {
        dir = 3;
        step = 0;
      }
      break;
     case 3:
      j -= stepSize;
      if (step == limit) {
        dir = 0;
        step = 0;
        limit++;
      }
      break;
    }
  }
#else
  /* Here we improve speed by checking first the most probable position
     then the search paths are most effectively cut. (0,0) is a simple start
  */
  unsigned int minerror = compareSubImg(Y_c, Y_p, field, linesize_c, linesize_p,
                                        md->fi.height, 1, 0, 0, UINT_MAX);
  // check all positions...
  for (i = -maxShift; i <= maxShift; i += stepSize) {
    for (j = -maxShift; j <= maxShift; j += stepSize) {
      if( i==0 && j==0 )
        continue; //no need to check this since already done
      unsigned int error = compareSubImg(Y_c, Y_p, field, linesize_c, linesize_p,
                                         md->fi.height, 1, i+offset.x, j+offset.y, minerror);
      if (error < minerror) {
	    previousMinError[previousMinErrorIndex] = minerror;
	    previousMinErrorIndex = (previousMinErrorIndex + 1)%minErrorCount;
        minerror = error;
        tx = i;
        ty = j;
      }
#ifdef STABVERBOSE
      fprintf(fieldLM, "%i %i %f\n", i, j, error);
#endif
    }
  }

#endif

  while(stepSize > 1) {// make fine grain check around the best match
    int txc = tx; // save the shifts
    int tyc = ty;
    int newStepSize = stepSize/2;
    int r = stepSize - newStepSize;
    for (i = txc - r; i <= txc + r; i += newStepSize) {
      for (j = tyc - r; j <= tyc + r; j += newStepSize) {
        if (i == txc && j == tyc) {
          continue; //no need to check this since already done
        }
        unsigned int error = compareSubImg(Y_c, Y_p, field, linesize_c, linesize_p,
                                           md->fi.height, 1, i+offset.x, j+offset.y, minerror);
#ifdef STABVERBOSE
        fprintf(fieldLM, "%i %i %f\n", i, j, error);
#endif

        //= TODO [CLEANUP]
//  	  if (md->frameNum == 5) {
//  		  fprintf(stderr, "{%i}-[%i,%i]-(%i)\n", fieldnum, i, j, error);
//  	  }
        if (error < minerror) {
		  previousMinError[previousMinErrorIndex] = minerror;
		  previousMinErrorTX[previousMinErrorIndex] = i;
		  previousMinErrorTY[previousMinErrorIndex] = j;
		  previousMinErrorIndex = (previousMinErrorIndex + 1)%MIN_ERROR_COUNT;
          minerror = error;
          tx = i;
          ty = j;
        }
      }
    }
    stepSize /= 2;
  }
#ifdef STABVERBOSE
  fclose(fieldLM);
  vs_log_msg(md->modName, "Minerror: %f\n", minerror);
#endif

  if (unlikely(fabs(tx) >= maxShift + stepSize - 1  ||
               fabs(ty) >= maxShift + stepSize)) {
#ifdef STABVERBOSE
    vs_log_msg(md->modName, "maximal shift ");
#endif
    lm.match =-1.0; // to be kicked out
    return lm;
  }
  lm.fieldLM = *field;
  lm.vector.x = tx + offset.x;
  lm.vector.y = ty + offset.y;
//  if (previousMinerror == -1) {
//	  lm.match = 255 * ((10000 - ((double)minerror / (field->size * field->size)))/10000);
//  } else {
//	  lm.match = ((double) previousMinerror - minerror) / (minerror/100);
//  }
//  lm.match = ((double)minerror / (field->size * field->size));

//  if (minerror > 10000) {
//	  lm.match = 100;
//  } else {
	  double uniqueness = 1;
	  double errorSameness;
	  double vectorDiff;
//	  int diff = 1;
	  unsigned int currentPreviousError;
	  int currentPreviousErrorTX;
	  int currentPreviousErrorTY;
	  int actualMinErrorCount = 0;

	  //=-- Determine actual stored error counts
	  for (int i = 0; i < MIN_ERROR_COUNT; i++) {
		  currentPreviousError = previousMinError[i];
		  actualMinErrorCount++;
	  }

	  //=-- Calculate uniqueness of match
	  for (int i = 0; i < MIN_ERROR_COUNT; i++) {
		  currentPreviousError = previousMinError[i];
		  currentPreviousErrorTX = previousMinErrorTX[i];
		  currentPreviousErrorTY = previousMinErrorTY[i];
		  if (currentPreviousError != NO_ERROR) {
			  errorSameness = (double) minerror / currentPreviousError;
			  vectorDiff = sqrt(pow(tx - currentPreviousErrorTX, 2) + pow(ty - currentPreviousErrorTY, 2))/ linesize_c;
			  uniqueness -= (vectorDiff * 100 * errorSameness) / actualMinErrorCount;
		  }
	  }
	  if (uniqueness < 0) {
		  uniqueness = 0.01;
	  }

	  lm.match = ((double)minerror/(field->size*field->size));
	  lm.match += lm.match / (uniqueness/2);
//  }

//  fprintf(stderr, "[%i->%i] Un: %f Min: %i Match: %f\n", md->frameNum ,fieldnum, uniqueness, minerror, lm.match);
  return lm;
}

/* calculates the optimal transformation for one field in Packed
 *   slower than the Planar version because it uses all three color channels
 */
LocalMotion calcFieldTransPacked(VSMotionDetect* md, VSMotionDetectFields* fs,
                                 const Field* field, int fieldnum) {
  int tx = 0;
  int ty = 0;
  uint8_t *I_c = md->curr.data[0], *I_p = md->prev.data[0];
  int width1 = md->curr.linesize[0]/3; // linesize in pixels
  int width2 = md->prev.linesize[0]/3; // linesize in pixels
  int i, j;
  int stepSize = fs->stepSize;
  int maxShift = fs->maxShift;

  Vec offset = { 0, 0};
  LocalMotion lm = null_localmotion();
  if(fs->useOffset){
    PreparedTransform pt = prepare_transform(&fs->offset, &md->fi);
    offset = transform_vec(&pt, (Vec*)field);
    // is the field still in the frame
    if(unlikely(offset.x-maxShift-stepSize < 0 || offset.x+maxShift+stepSize >= md->fi.width ||
                offset.y-maxShift-stepSize < 0 || offset.y+maxShift+stepSize >= md->fi.height)){
      lm.match=-1;
      return lm;
    }
  }

  /* Here we improve speed by checking first the most probable position
     then the search paths are most effectively cut. (0,0) is a simple start
  */
  unsigned int minerror = compareSubImg(I_c, I_p, field, width1, width2, md->fi.height,
                                        3, offset.x, offset.y, UINT_MAX);
  // check all positions...
  for (i = -maxShift; i <= maxShift; i += stepSize) {
    for (j = -maxShift; j <= maxShift; j += stepSize) {
      if( i==0 && j==0 )
        continue; //no need to check this since already done
      unsigned int error = compareSubImg(I_c, I_p, field, width1, width2,
                                         md->fi.height, 3, i + offset.x, j + offset.y, minerror);
      if (error < minerror) {
        minerror = error;
        tx = i;
        ty = j;
      }
    }
  }
  if (stepSize > 1) { // make fine grain check around the best match
    int txc = tx; // save the shifts
    int tyc = ty;
    int r = stepSize - 1;
    for (i = txc - r; i <= txc + r; i += 1) {
      for (j = tyc - r; j <= tyc + r; j += 1) {
        if (i == txc && j == tyc)
          continue; //no need to check this since already done
        unsigned int error = compareSubImg(I_c, I_p, field, width1, width2,
                                           md->fi.height, 3, i + offset.x, j + offset.y, minerror);
        if (error < minerror) {
          minerror = error;
          tx = i;
          ty = j;
        }
      }
    }
  }

  if (fabs(tx) >= maxShift + stepSize - 1 || fabs(ty) >= maxShift + stepSize - 1) {
#ifdef STABVERBOSE
    vs_log_msg(md->modName, "maximal shift ");
#endif
    lm.match = -1;
    return lm;
  }
  lm.fieldLM = *field;
  lm.vector.x = tx + offset.x;
  lm.vector.y = ty + offset.y;
  lm.match = ((double)minerror)/(field->size*field->size);
  return lm;
}

/* compares contrast_idx structures respect to the contrast
   (for sort function)
*/
int contrastComparator(const void *ci1, const void* ci2) {
  double a = ((ContrastContainer*) ci1)->contrast;
  double b = ((ContrastContainer*) ci2)->contrast;
  return a < b ? 1 : (a > b ? -1 : 0);
}

/* Select only the best 'maxfields' fields. Best is defined here as fields with the highest contrast.
 *  1. Calculate contrasts for all fields
 *  2. Select {fs->maxFields} fields from each segment of the frame
 *
 *  TODO [Low] Is the following comment still relevant?
 *  We may simplify here by using random. People want high quality, so typically we use all.
 */
// TODO [DEBUGGING] Select fields
VSVector selectFields(VSMotionDetect* md, VSMotionDetectFields* fieldsContainer,
		contrastSubImgFunc contrastfunc) {

	//=-- Declare and initialize variables
	VSVector selectedFields;
	vs_vector_init(&selectedFields, fieldsContainer->fieldNum);

	//=-- Create containers for the contrast values. The second one is used in a slightly convoluted method of sorting and selecting fields,
	ContrastContainer *contrastByField = (ContrastContainer*) vs_malloc(
			sizeof(ContrastContainer) * fieldsContainer->fieldNum);
	ContrastContainer *contrastByFieldSortable = (ContrastContainer*) vs_malloc(
			sizeof(ContrastContainer) * fieldsContainer->fieldNum);

	//=-- Determine number of segments, segment length and how many fields we initially wish to select per segment
	int numberOfSegments = (fieldsContainer->fieldRows + 1);
	int fieldsPerSegment = fieldsContainer->fieldNum
			/ (fieldsContainer->fieldRows + 1) + 1;
	int maxFieldsPerSegment = fieldsContainer->maxFields / numberOfSegments;

	//=-- Calculate contrast for each field
	// #pragma omp parallel for shared(ci,md) no speedup because to short
	int i;
	for (i = 0; i < fieldsContainer->fieldNum; i++) {
		contrastByField[i].contrast = contrastfunc(md,
				&fieldsContainer->fields[i]);
		contrastByField[i].index = i;

//		fprintf(stderr, "%i %lf\n", contrastByField[i].index, contrastByField[i].contrast);
		if (contrastByField[i].contrast < fieldsContainer->contrastThreshold) {
			contrastByField[i].contrast = 0;
		}
	}

	//=-- Duplicate contrasts
	memcpy(contrastByFieldSortable, contrastByField,
			sizeof(ContrastContainer) * fieldsContainer->fieldNum);

	//=-- For each segment, retrieve the fields with the best contrast, up to maxFieldsPerSegment and fieldsContainer->maxFields total
	int relativeFieldIndex;
	int segmentNumber;
	for (segmentNumber = 0; segmentNumber < numberOfSegments; segmentNumber++) {

		//=- Process only fields of this segment
		int fieldStartIndex = fieldsPerSegment * segmentNumber;
		int fieldEndindex = fieldStartIndex + fieldsPerSegment - 1;
		//printf("Segment: %i: %i-%i\n", i, startIndex, endIndex);

		//=- Limit end index
		fieldEndindex =
				fieldEndindex > fieldsContainer->fieldNum-1 ?
						fieldsContainer->fieldNum-1 : fieldEndindex;

		//=- Sort by contrast within segment
		qsort(contrastByFieldSortable + fieldStartIndex,
				fieldEndindex - fieldStartIndex, sizeof(ContrastContainer),
				contrastComparator);

		//=- Start selecting best fields
		int selectedFieldCount = 0;
		int absoluteFieldIndex = fieldStartIndex;
		relativeFieldIndex = 0;

		while (selectedFieldCount < maxFieldsPerSegment
				&& absoluteFieldIndex < fieldEndindex) {

			absoluteFieldIndex = fieldStartIndex + relativeFieldIndex;

			//		  fprintf(stderr, "Processing absolute index %i of %i", absoluteFieldIndex, fieldsContainer->fieldNum);
//			       fprintf(stderr, "%i %lf\n", contrastByFieldSortable[absoluteFieldIndex].index,
//			                          contrastByFieldSortable[absoluteFieldIndex].contrast);
			if (contrastByFieldSortable[absoluteFieldIndex].contrast > 0) {
				ContrastContainer* currentContrastContainer =
						&contrastByField[contrastByFieldSortable[absoluteFieldIndex].index];
				vs_vector_append_dup(&selectedFields, currentContrastContainer,
						sizeof(ContrastContainer));

				//= Make field as not eligible for next phase
				contrastByFieldSortable[absoluteFieldIndex].contrast = 0;

				selectedFieldCount++;
			}

			relativeFieldIndex++;
		}
	}

	//=-- Check whether enough fields are selected
	// printf("Phase2: %i\n", vs_list_size(selectedFields));
	int remaining = fieldsContainer->maxFields
			- vs_vector_size(&selectedFields);
	int j;
	if (remaining > 0) {
		//=- Not enough fields were selected, so select the fields with the best contrast over all segments
		qsort(contrastByFieldSortable, fieldsContainer->fieldNum,
				sizeof(ContrastContainer), contrastComparator);
		for (j = 0; j < remaining; j++) {
			if (contrastByFieldSortable[j].contrast > 0) {
				ContrastContainer* currentContrastContainer =
						&contrastByFieldSortable[j];
				vs_vector_append_dup(&selectedFields, currentContrastContainer,
						sizeof(ContrastContainer));
			}
		}
	}

	//=-- Cleanup and return
	// printf("Ende: %i\n", vs_list_size(selectedFields));
	vs_free(contrastByField);
	vs_free(contrastByFieldSortable);

	return selectedFields;
}

/* tries to register current frame onto previous frame.
 *   Algorithm:
 *   discards fields with low contrast
 *   select maxfields fields according to their contrast
 *   check theses fields for vertical and horizontal transformation
 *   use minimal difference of all possible positions
 */
LocalMotions calcTransFields(VSMotionDetect* md,
                             VSMotionDetectFields* fields,
                             calcFieldTransFunc fieldfunc,
                             contrastSubImgFunc contrastfunc) {
  LocalMotions localmotions;
  vs_vector_init(&localmotions,fields->maxFields);
#ifdef STABVERBOSE
  FILE *file = NULL;
  char buffer[32];
  vs_snprintf(buffer, sizeof(buffer), "k%04i.dat", md->frameNum);
  file = fopen(buffer, "w");
  fprintf(file, "# plot \"%s\" w l, \"\" every 2:1:0\n", buffer);
#endif

  VSVector goodflds = selectFields(md, fields, contrastfunc);
  // use all "good" fields and calculate optimal match to previous frame
#ifdef USE_OMP
#pragma omp parallel for shared(goodflds, md, localmotions, fs) // does not bring speedup
#endif
  for(int index=0; index < vs_vector_size(&goodflds); index++){
    int i = ((ContrastContainer*)vs_vector_get(&goodflds,index))->index;
    LocalMotion m;
    m = fieldfunc(md, fields, &fields->fields[i], i); // e.g. calcFieldTransPlanar
    if(m.match >= 0){
      m.contrast = ((ContrastContainer*)vs_vector_get(&goodflds,index))->contrast;
#ifdef STABVERBOSE
      fprintf(file, "%i %i\n%f %f %f %f\n \n\n", m.fieldLM.x, m.fieldLM.y,
              m.fieldLM.x + m.vector.x, m.fieldLM.y + m.vector.y, m.match, m.contrast);
#endif
      vs_vector_append_dup(&localmotions, &m, sizeof(LocalMotion));
    }
  }
  vs_vector_del(&goodflds);

#ifdef STABVERBOSE
  fclose(file);
#endif
  return localmotions;
}





/** draws the field scanning area */
void drawFieldScanArea(VSMotionDetect* md, const LocalMotion* lm, int maxShift) {
  if (md->fi.pFormat > PF_PACKED)
    return;
  drawRectangle(md->currorig.data[0], md->currorig.linesize[0], md->fi.height, 1, lm->fieldLM.x, lm->fieldLM.y,
                lm->fieldLM.size + 2 * maxShift, lm->fieldLM.size + 2 * maxShift, 80);
}

/** draws the field */
void drawField(VSMotionDetect* md, const LocalMotion* lm, short box) {
  if (md->fi.pFormat > PF_PACKED)
    return;
//  fprintf(stderr, "Match %f\n", lm->match);
  if(box) {
    drawBox(md->currorig.data[0], md->currorig.linesize[0], md->fi.height, 1,
            lm->fieldLM.x, lm->fieldLM.y, lm->fieldLM.size, lm->fieldLM.size, VS_MIN(255,lm->match*30) /*lm->match >100 ? 100 : 40 */);
  }
  else {
    drawRectangle(md->currorig.data[0], md->currorig.linesize[0], md->fi.height, 1,
                  lm->fieldLM.x, lm->fieldLM.y, lm->fieldLM.size, lm->fieldLM.size, VS_MIN(255,lm->match*30) /*lm->match >100 ? 100 : 40 */);
  }
}

/** draws the transform data of this field */
void drawFieldTrans(VSMotionDetect* motionDetect, const LocalMotion* localMotion, int color) {
  if (motionDetect->fi.pFormat > PF_PACKED)
    return;
  Vec end = sub_vec(field_to_vec(localMotion->fieldLM),localMotion->vector);
  drawBox(motionDetect->currorig.data[0], motionDetect->currorig.linesize[0], motionDetect->fi.height, 1,
          localMotion->fieldLM.x, localMotion->fieldLM.y, 5, 5, 0); // draw center
  drawBox(motionDetect->currorig.data[0], motionDetect->currorig.linesize[0], motionDetect->fi.height, 1,
          localMotion->fieldLM.x - localMotion->vector.x, localMotion->fieldLM.y - localMotion->vector.y, 5, 5, 250); // draw translation
  drawLine(motionDetect->currorig.data[0], motionDetect->currorig.linesize[0],  motionDetect->fi.height, 1,
           (Vec*)&localMotion->fieldLM, &end, 3, color);

}

/**
 * draws a box at the given position x,y (center) in the given color
 (the same for all channels)
*/
void drawBox(unsigned char* I, int width, int height, int bytesPerPixel, int x,
       int y, int sizex, int sizey, unsigned char color) {

  unsigned char* p = NULL;
  int j, k;
  p = I + ((x - sizex / 2) + (y - sizey / 2) * width) * bytesPerPixel;
  for (j = 0; j < sizey; j++) {
    for (k = 0; k < sizex * bytesPerPixel; k++) {
      *p = color;
      p++;
    }
    p += (width - sizex) * bytesPerPixel;
  }
}

/**
 * draws a rectangle (not filled) at the given position x,y (center) in the given color
 at the first channel
*/
void drawRectangle(unsigned char* I, int width, int height, int bytesPerPixel, int x,
                   int y, int sizex, int sizey, unsigned char color) {

  unsigned char* p;
  int k;
  p = I + ((x - sizex / 2) + (y - sizey / 2) * width) * bytesPerPixel;
  for (k = 0; k < sizex; k++) { *p = color; p+= bytesPerPixel; } // upper line
  p = I + ((x - sizex / 2) + (y + sizey / 2) * width) * bytesPerPixel;
  for (k = 0; k < sizex; k++) { *p = color; p+= bytesPerPixel; } // lower line
  p = I + ((x - sizex / 2) + (y - sizey / 2) * width) * bytesPerPixel;
  for (k = 0; k < sizey; k++) { *p = color; p+= width * bytesPerPixel; } // left line
  p = I + ((x + sizex / 2) + (y - sizey / 2) * width) * bytesPerPixel;
  for (k = 0; k < sizey; k++) { *p = color; p+= width * bytesPerPixel; } // right line
}

/**
 * draws a line from a to b with given thickness(not filled) at the given position x,y (center) in the given color
 at the first channel
*/
void drawLine(unsigned char* I, int width, int height, int bytesPerPixel,
              Vec* a, Vec* b, int thickness, unsigned char color) {
  unsigned char* p;
  Vec div = sub_vec(*b,*a);
  if(div.y==0){ // horizontal line
    if(div.x<0) {*a=*b; div.x*=-1;}
    for(int r=-thickness/2; r<=thickness/2; r++){
      p = I + ((a->x) + (a->y+r) * width) * bytesPerPixel;
      for (int k = 0; k <= div.x; k++) { *p = color; p+= bytesPerPixel; }
    }
  }else{
    if(div.x==0){ // vertical line
      if(div.y<0) {*a=*b; div.y*=-1;}
      for(int r=-thickness/2; r<=thickness/2; r++){
        p = I + ((a->x+r) + (a->y) * width) * bytesPerPixel;
        for (int k = 0; k <= div.y; k++) { *p = color; p+= width * bytesPerPixel; }
      }
    }else{
      double m = (double)div.x/(double)div.y;
      int horlen = thickness + fabs(m);
      for( int c=0; c<= abs(div.y); c++){
        int dy = div.y<0 ? -c : c;
        int x = a->x + m*dy - horlen/2;
        p = I + (x + (a->y+dy) * width) * bytesPerPixel;
        for( int k=0; k<= horlen; k++){ *p = color; p+= bytesPerPixel; }
      }
    }
  }
}


// void addTrans(VSMotionDetect* md, VSTransform sl) {
//   if (!md->transs) {
//     md->transs = vs_list_new(0);
//   }
//   vs_list_append_dup(md->transs, &sl, sizeof(sl));
// }

// VSTransform getLastVSTransform(VSMotionDetect* md){
//   if (!md->transs || !md->transs->head) {
//     return null_transform();
//   }
//   return *((VSTransform*)md->transs->tail);
// }


//#ifdef TESTING
/// plain C implementation of compareSubImg (without ORC)
unsigned int compareSubImg_thr(unsigned char* const I1, unsigned char* const I2,
                               const Field* field, int width1, int width2, int height,
           int bytesPerPixel, int d_x, int d_y,
           unsigned int threshold) {

  int k, j;
  unsigned char* p1 = NULL;
  unsigned char* p2 = NULL;
  int s2 = field->size / 2;
  unsigned int sum = 0;

  int baseX = field->x - s2;
  int baseY = field->y - s2;

  p1 = I1 + (baseX + baseY * width1) * bytesPerPixel;
  p2 = I2 + ((baseX + d_x) + (baseY + d_y) * width2) * bytesPerPixel;
  for (j = 0; j < field->size; j++) {
    for (k = 0; k < (field->size * bytesPerPixel)-1; k++) {

      int p1Value = (int) *p1;
      int p2Value = (int) *p2;

      sum += abs(p1Value - p2Value);

      p1++;
      p2++;
    }
    if( sum > threshold) // no need to calculate any longer: worse than the best match
      break;
    p1 += (width1 - field->size) * bytesPerPixel;
    p2 += (width2 - field->size) * bytesPerPixel;
  }
  return sum;
}

void initContrastLookup() {
	if (contrastLookupInitialized == 0) {
		double newValue = 0;
		double c = (100.0 + contrast) / 100.0;
		c *= c;

		for (int i = 0; i < 256; i++) {
			newValue = (double) i;
			newValue /= 255.0;
			newValue -= 0.5;
			newValue *= c;
			newValue += 0.5;
			newValue *= 255;

			if (newValue < 0)
				newValue = 0;
			if (newValue > 255)
				newValue = 255;
			contrastLookup[i] = (int) newValue;
		}
	}
}

void initDerivativeKernel() {
	if (derivativeKernelInitialized == 0) {
		// Kernel looks something like this:
		// [ 0 , 3 , 2 , 1]
		// [ 3 , 2 , 1 , 0]
		// [ 2 , 1 , 0 , 0]
		// [ 1 , 0 , 0 , 0]

		float rowValue;
		float value;
		float total = 0;
		for (int i = 0; i < derivativeKernelSize; i++) {
			rowValue = derivativeKernelSize - i;
			for (int j = 0; j < derivativeKernelSize; j++) {
				value = VS_MAX(0, rowValue - j);
				derivativeKernel[i][j] = value;
				total += value;
			}
		}
		derivativeKernel[0][0] = 0;

		//=-- Normalize kernel
		for (int i = 0; i < derivativeKernelSize; i++) {
			for (int j = 0; j < derivativeKernelSize; j++) {
				derivativeKernel[i][j] = derivativeKernel[i][j] / total;
			}
		}
	}
}

// TODO [CUSTOM] Derivative
void calculateDerivativeImage(unsigned char* dest, const unsigned char* src,
		VSFrame* buffer, const VSFrameInfo* fi, int destFrameWidth,
		int srcFrameWidth) {

	//=-- Declare and init some variables
	int i, j, k, l;
	double accumulator;
	const unsigned char *sourcePixelPointer;
	unsigned char *destPixelPointer;
	int targetPixelValue;
	int localbuffer = 0;

	//=-- Create buffer if necessary
	VSFrame buf;
	if (buffer == 0) {
		vsFrameAllocate(&buf, fi);
		localbuffer = 1;
	} else {
		buf = *buffer;
	}

	//=-- Generate contrast lookup table
	initContrastLookup();

	//=-- Init kernel
	initDerivativeKernel();

	//=-- Calculate derivative image and store it in dest
	float kernelWeight;
	for (k = 0; k < fi->height - derivativeKernelSize; k++) {
		for (l = 0; l < fi->width - derivativeKernelSize; l++) {
			accumulator = 0;
			int offset = k * srcFrameWidth + l;
			sourcePixelPointer = src + offset;
			destPixelPointer = dest + offset;
			unsigned char sourceValue = (*sourcePixelPointer);

			//=-- Process comparison pixels
			for (i = 0; i < derivativeKernelSize; i++) {
				for (j = 0; j < derivativeKernelSize; j++) {
					kernelWeight = derivativeKernel[i][j];
					if (kernelWeight != 0) {
						int kernelOffset = i * srcFrameWidth + j;
						const unsigned char* comparisonPixelPointer =
								sourcePixelPointer + kernelOffset;
						unsigned char comparisonValue =
								(*comparisonPixelPointer);
						int delta = (sourceValue - comparisonValue);
						accumulator += kernelWeight * delta;
					}
				}
			}

			//=-- Shift value to fit in 0-255 range
			accumulator = accumulator / 2 + 128;
//			fprintf(stderr, "Acc: %f\n", accumulator);

			targetPixelValue = (int) accumulator;

			//=-- Apply contrast
			targetPixelValue = contrastLookup[targetPixelValue];

			//=-- Write value
			(*destPixelPointer) = targetPixelValue;

		}
	}

	//=-- Clean buffer if local
	if (localbuffer == 1) {
		vsFrameFree(&buf);
	}
}


/*
 * Local variables:
 *   c-file-style: "stroustrup"
 *   c-file-offsets: ((case-label . *) (statement-case-intro . *))
 *   indent-tabs-mode: nil
 *   tab-width:  2
 *   c-basic-offset: 2 t
 * End:
 *
 */
