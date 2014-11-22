/*
 *  transform.c
 *
 *  Copyright (C) Georg Martius - June 2007 - 2011
 *   georg dot martius at web dot de
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

#include "transform.h"
#include "transform_internal.h"
#include "transformtype_operations.h"

#include "transformfixedpoint.h"
#ifdef TESTING
#include "transformfloat.h"
#endif

#include <math.h>
#include <libgen.h>
#include <string.h>

const char* interpol_type_names[5] = {"No (0)", "Linear (1)", "Bi-Linear (2)",
                                      "Bi-Cubic (3)"};

const char* getInterpolationTypeName(VSInterpolType type){
  if (type >= VS_Zero && type < VS_NBInterPolTypes)
    return interpol_type_names[(int) type];
  else
    return "unknown";
}

// default initialization: attention the ffmpeg filter cannot call it
VSTransformConfig vsTransformGetDefaultConfig(const char* modName){
  VSTransformConfig conf;
  /* Options */
  conf.maxShift           = -1;
  conf.maxAngle           = -1;
  conf.crop               = VSKeepBorder;
  conf.relative           = 1;
  conf.invert             = 0;
  conf.smoothing          = 15;
  conf.zoom               = 0;
  conf.optZoom            = 1;
  conf.zoomSpeed          = 0.25;
  conf.interpolType       = VS_BiLinear;
  conf.verbose            = 0;
  conf.modName            = modName;
  conf.simpleMotionCalculation = 0;
  conf.storeTransforms    = 0;
  conf.smoothZoom         = 0;
  conf.camPathAlgo        = VSOptimalL1;
  return conf;
}

void vsTransformGetConfig(VSTransformConfig* conf, const VSTransformData* td){
  if(td && conf)
    *conf = td->conf;
}

const VSFrameInfo* vsTransformGetSrcFrameInfo(const VSTransformData* td){
  return &td->fiSrc;
}

const VSFrameInfo* vsTransformGetDestFrameInfo(const VSTransformData* td){
  return &td->fiDest;
}


int vsTransformDataInit(VSTransformData* td, const VSTransformConfig* conf,
                        const VSFrameInfo* fi_src, const VSFrameInfo* fi_dest){
  td->conf = *conf;

  td->fiSrc = *fi_src;
  td->fiDest = *fi_dest;

  vsFrameNull(&td->src);
  td->srcMalloced = 0;

  vsFrameNull(&td->destbuf);
  vsFrameNull(&td->dest);

  if (td->conf.maxShift > td->fiDest.width/2)
    td->conf.maxShift = td->fiDest.width/2;
  if (td->conf.maxShift > td->fiDest.height/2)
    td->conf.maxShift = td->fiDest.height/2;

  td->conf.interpolType = VS_MAX(VS_MIN(td->conf.interpolType,VS_BiCubic),VS_Zero);

  // not yet implemented
  if(td->conf.camPathAlgo==VSOptimalL1) td->conf.camPathAlgo=VSGaussian;

  switch(td->conf.interpolType){
   case VS_Zero:     td->interpolate = &interpolateZero; break;
   case VS_Linear:   td->interpolate = &interpolateLin; break;
   case VS_BiLinear: td->interpolate = &interpolateBiLin; break;
   case VS_BiCubic:  td->interpolate = &interpolateBiCub; break;
   default: td->interpolate = &interpolateBiLin;
  }
#ifdef TESTING
  switch(td->conf.interpolType){
   case VS_Zero:     td->_FLT(interpolate) = &_FLT(interpolateZero); break;
   case VS_Linear:   td->_FLT(interpolate) = &_FLT(interpolateLin); break;
   case VS_BiLinear: td->_FLT(interpolate) = &_FLT(interpolateBiLin); break;
   case VS_BiCubic:  td->_FLT(interpolate) = &_FLT(interpolateBiCub); break;
   default: td->_FLT(interpolate)          = &_FLT(interpolateBiLin);
  }

#endif
  return VS_OK;
}

void vsTransformDataCleanup(VSTransformData* td){
  if (td->srcMalloced && !vsFrameIsNull(&td->src)) {
    vsFrameFree(&td->src);
  }
  if (td->conf.crop == VSKeepBorder && !vsFrameIsNull(&td->destbuf)) {
    vsFrameFree(&td->destbuf);
  }
}

int vsTransformPrepare(VSTransformData* td, const VSFrame* src, VSFrame* dest){
  // we first copy the frame to td->src and then overwrite the destination
  // with the transformed version
  td->dest = *dest;
  if(src==dest || td->srcMalloced){ // in place operation: we have to copy the src first
    if(vsFrameIsNull(&td->src)) {
      vsFrameAllocate(&td->src,&td->fiSrc);
      td->srcMalloced = 1;
    }
    if (vsFrameIsNull(&td->src)) {
      vs_log_error(td->conf.modName, "vs_malloc failed\n");
      return VS_ERROR;
    }
    vsFrameCopy(&td->src, src, &td->fiSrc);
  }else{ // otherwise no copy needed
    td->src=*src;
  }
  if (td->conf.crop == VSKeepBorder) {
    if(vsFrameIsNull(&td->destbuf)) {
      // if we keep the borders, we need a second buffer to store
      //  the previous stabilized frame, so we use destbuf
      vsFrameAllocate(&td->destbuf,&td->fiDest);
      if (vsFrameIsNull(&td->destbuf)) {
        vs_log_error(td->conf.modName, "vs_malloc failed\n");
        return VS_ERROR;
      }
      // if we keep borders, save first frame into the background buffer (destbuf)
      vsFrameCopy(&td->destbuf, src, &td->fiSrc); // here we have to take care
    }
  }else{ // otherwise we directly operate on the destination
    td->destbuf = *dest;
  }
  return VS_OK;
}

int vsDoTransform(VSTransformData* td, VSTransform t){
  if (td->fiSrc.pFormat < PF_PACKED)
    return transformPlanar(td, t);
  else
    return transformPacked(td, t);
}


int vsTransformFinish(VSTransformData* td){
  if(td->conf.crop == VSKeepBorder){
    // we have to store our result to video buffer
    // note: destbuf stores stabilized frame to be the default for next frame
    vsFrameCopy(&td->dest, &td->destbuf, &td->fiSrc);
  }
  return VS_OK;
}


VSTransform vsGetNextTransform(const VSTransformData* td, VSTransformationsContainer* trans){
  if(trans->len <=0 ) return null_transform();
  if (trans->current >= trans->len) {
    trans->current = trans->len;
    if(!trans->warned_end)
      vs_log_warn(td->conf.modName, "not enough transforms found, use last transformation!\n");
    trans->warned_end = 1;
  }else{
    trans->current++;
  }
  return trans->ts[trans->current-1];
}

void vsTransformationsInit(VSTransformationsContainer* trans){
  trans->ts = 0;
  trans->len = 0;
  trans->current = 0;
  trans->warned_end = 0;
}

void vsTransformationsCleanup(VSTransformationsContainer* trans){
  if (trans->ts) {
    vs_free(trans->ts);
    trans->ts = NULL;
  }
  trans->len=0;
}

/*
 *  This is actually the core algorithm for canceling the jiggle in the
 *  movie. We have different implementations which are patched here.
 */
int cameraPathOptimization(VSTransformData* td,
		VSTransformationsContainer* trans) {
	switch (td->conf.camPathAlgo) {
	case VSAvg:
//		fprintf(stderr, "Doing AVG transform\n");
		return cameraPathAvg(td, trans);
		break;

	case VSOptimalL1: // not yet implenented
	case VSGaussian:
//		fprintf(stderr, "Doing Gaussian transform\n");
		return cameraPathGaussian(td, trans);
		break;

	}
	return VS_ERROR;
}

/*
 *  We perform a low-pass filter on the camera path.
 *  This supports slow camera movement, but in a smooth fashion.
 *  Here we use gaussian filter (gaussian kernel) lowpass filter
 */
// TODO [DEBUGGING] Camera path smoothing
int cameraPathGaussian(VSTransformData* td, VSTransformationsContainer* transformsContainer) {

	VSTransform* transforms = transformsContainer->ts;
	if (transformsContainer->len < 1)
		return VS_ERROR;
	if (td->conf.verbose & VS_DEBUG) {
		vs_log_msg(td->conf.modName, "Preprocess transforms:");
	}


	/* relative to absolute (integrate transformations) */
	// TODO The difference between absolute and relative isn't quite clear.
	if (td->conf.relative) {
		VSTransform t = transforms[0];
		for (int i = 1; i < transformsContainer->len; i++) {
			transforms[i] = add_transforms(&transforms[i], &t);
			t = transforms[i];
		}
	}

	if (td->conf.smoothing > 0) {
		//=-- Create copy of transforms container to work with
		VSTransform* transformsContainerCopy = vs_malloc(
				sizeof(VSTransform) * transformsContainer->len);
		memcpy(transformsContainerCopy, transforms, sizeof(VSTransform) * transformsContainer->len);

		//=-- TODO [CUSTOM] Set all transforms for frames with extra == 1 to linear difference between previous good and next good frame
		VSTransform* transformPointer;
		VSTransform* nextTransformPointer;
		VSTransform* endTransformPointer;
		VSTransform* startTransformPointer = &transforms[0];
		for (int i = 1; i < transformsContainer->len; i++) {
			if (i!=0) {
				startTransformPointer = &transforms[i-1];
			}
			transformPointer = &transforms[i];
			if (transformPointer->extra == 1) {
				//=-- Retrieve end transForm and distance
				int consecutiveBadFrames = 1;
				endTransformPointer = transformPointer;
				int MAX_BAD_CONSECUTIVE_FRAMES_CORRECTION = 5;

				while (consecutiveBadFrames < MAX_BAD_CONSECUTIVE_FRAMES_CORRECTION && consecutiveBadFrames + i < transformsContainer->len) {
					int index = i+consecutiveBadFrames;
					nextTransformPointer = &transforms[index];
					endTransformPointer = nextTransformPointer;

					//=-- Stop if we find a good end frame
					if (endTransformPointer->extra !=1 ) {
						break;
					}
					consecutiveBadFrames++;
				}

				//=-- Only process if a good end frame was found
				if (endTransformPointer->extra !=1 ) {
					//=-- Calculate transform delta between good frames
					VSTransform deltaTransform;
					memcpy(&deltaTransform, endTransformPointer,sizeof(VSTransform));
					deltaTransform = sub_transforms(&deltaTransform, startTransformPointer);

	//				fprintf(stderr,"Start: [%i] - x:%5lf y:%5lf r:%5lf e:%i \n",i, startTransformPointer->x,startTransformPointer->y, startTransformPointer->rotate, startTransformPointer->extra);
	//				fprintf(stderr,"End: [%i] - x:%5lf y:%5lf r:%5lf e:%i \n",i, endTransformPointer->x,endTransformPointer->y, endTransformPointer->rotate, endTransformPointer->extra);
	//				fprintf(stderr,"Delta: [%i] - x:%5lf y:%5lf r:%5lf e:%i \n",i, deltaTransform.x,deltaTransform.y, deltaTransform.rotate, deltaTransform.extra);

					//=-- Update intermediate transforms
					int relativeIndex = 1;
					while (relativeIndex <= consecutiveBadFrames) {
						int currentIndex = i+relativeIndex-1;
//						fprintf(stderr,"Previous: [%i] - x:%5lf y:%5lf r:%5lf z:%5lf\n",i, transformPointer->x,transformPointer->y, transformPointer->rotate, transformPointer->zoom);

						double multFactor = 0.5;
						VSTransform multTransform = mult_transform(&deltaTransform, multFactor);
						transforms[currentIndex] = add_transforms(startTransformPointer, &multTransform);
						transformPointer = &transforms[currentIndex];
						transformPointer->extra = 0;

//						transformPointer->x = startTransformPointer->x + deltaTransform.x * multFactor;
//						transformPointer->y = startTransformPointer->y + deltaTransform.y * multFactor;
//						transformPointer->rotate = startTransformPointer->rotate + deltaTransform.rotate * multFactor;
//						transformPointer->barrel = startTransformPointer->barrel + deltaTransform.barrel * multFactor;
//						transformPointer->rshutter = startTransformPointer->rshutter + deltaTransform.rshutter * multFactor;
//						transformPointer->zoom = startTransformPointer->zoom + deltaTransform.zoom * multFactor;
//						transformPointer->extra = 0;
//						fprintf(stderr,"Override: [%i] - x:%5lf y:%5lf r:%5lf z:%5lf\n",i, transformPointer->x,transformPointer->y, transformPointer->rotate, transformPointer->zoom);

						relativeIndex++;
					}

					//=-- Advance i the additional amount of bad frames we have processed, besides the current
					i += consecutiveBadFrames - 1;
				}

			}
		}


		//=-- Initialize variables
		int smoothingOneSidedCount = td->conf.smoothing;
		int smoothingTwoSidedCount = td->conf.smoothing * 2 + 1;
		double sigma2 = sqr(smoothingOneSidedCount / 2.0);

		//=-- Initialize 1D gaussian kernel
		VSArray kernel = vs_array_new(smoothingTwoSidedCount);
		for (int i = 0; i <= smoothingOneSidedCount; i++) {
			kernel.dat[i] = kernel.dat[smoothingTwoSidedCount - i - 1] = exp(
					-sqr(i - smoothingOneSidedCount) / sigma2);
		}
//		vs_array_print(kernel, stdout);

		//=-- Process all frames
		for (int i = 0; i < transformsContainer->len; i++) {
//			currentTransform = transforms[i];
//			fprintf(stderr,"Pre: [%i] - x:%5lf y:%5lf r:%5lf e:%i\n",i, currentTransform.x,currentTransform.y, currentTransform.rotate, currentTransform.extra );

			// make a convolution:
			double weightsum = 0;
			VSTransform avg = null_transform();
//			fprintf(stderr, "Doing smoothing for frame %i/%i \n", i, transformsContainer->len);

			//=- Process all frames within the smoothing bandwidth
			for (int relativeFrameIndex = 0;
					relativeFrameIndex < smoothingTwoSidedCount;
					relativeFrameIndex++) {
				int absoluteFrameIndex = i + relativeFrameIndex - smoothingOneSidedCount;

				//=- Check whether frame index is in all frames range
				if (absoluteFrameIndex >= 0 && absoluteFrameIndex < transformsContainer->len) {

					//=- Dont use bad frames or scene cuts for smoothing
					if (transformsContainerCopy[absoluteFrameIndex].extra == 1) {
						continue;

//						// deal with scene cuts or bad frames
//						fprintf(stderr, "Encountered bad frame [%i]-(%i) \n", absoluteFrameIndex, relativeFrameIndex);
//						if (relativeFrameIndex < smoothingOneSidedCount) { // in the past of our frame: ignore everthing before
//							avg = null_transform();
//							weightsum = 0;
//							continue;
//						}
//						else if (relativeFrameIndex
//								== smoothingOneSidedCount) {
//							//current frame or in future: stop here
//							//for current frame: ignore completely
//							weightsum = 0;
//
//							break;
//						}
					}
					weightsum += kernel.dat[relativeFrameIndex];
					avg = add_transforms_(avg,
							mult_transform(&transformsContainerCopy[absoluteFrameIndex],
									kernel.dat[relativeFrameIndex]));
				}
			}

			if (weightsum > 0) {
				avg = mult_transform(&avg, 1.0 / weightsum);

				// high frequency must be transformed away
				transforms[i] = sub_transforms(&transforms[i], &avg);
			}

			if (td->conf.verbose & VS_DEBUG) {
				vs_log_msg(td->conf.modName,
						" avg: %5lf, %5lf, %5lf extra: %i weightsum %5lf",
						avg.x, avg.y, avg.rotate, transforms[i].extra, weightsum);
			}
		}

//		VSTransform currentTransform;
//		for (int i = 0; i < transformsContainer->len; i++) {
//			currentTransform = transforms[i];
//			fprintf(stderr,"Post: [%i] - x:%5lf y:%5lf r:%5lf e:%i\n",i, currentTransform.x,currentTransform.y, currentTransform.rotate, currentTransform.extra );
//		}

	}

	return VS_OK;
}

/*
 *  We perform a low-pass filter in terms of transformations.
 *  This supports slow camera movement (low frequency), but in a smooth fasion.
 *  Here a simple average based filter
 */
int cameraPathAvg(VSTransformData* td, VSTransformationsContainer* trans){
  VSTransform* ts = trans->ts;

  if (trans->len < 1)
    return VS_ERROR;
  if (td->conf.verbose & VS_DEBUG) {
   vs_log_msg(td->conf.modName, "Preprocess transforms:");
  }
  if (td->conf.smoothing>0) {
    /* smoothing */
    VSTransform* ts2 = vs_malloc(sizeof(VSTransform) * trans->len);
    memcpy(ts2, ts, sizeof(VSTransform) * trans->len);

    /*  we will do a sliding average with minimal update
     *   \hat x_{n/2} = x_1+x_2 + .. + x_n
     *   \hat x_{n/2+1} = x_2+x_3 + .. + x_{n+1} = x_{n/2} - x_1 + x_{n+1}
     *   avg = \hat x / n
     */
    int s = td->conf.smoothing * 2 + 1;
    VSTransform null = null_transform();
    /* avg is the average over [-smoothing, smoothing] transforms
       around the current point */
    VSTransform avg;
    /* avg2 is a sliding average over the filtered signal! (only to past)
     *  with smoothing * 2 horizon to kill offsets */
    VSTransform avg2 = null_transform();
    double tau = 1.0/(2 * s);
    /* initialise sliding sum with hypothetic sum centered around
     * -1st element. We have two choices:
     * a) assume the camera is not moving at the beginning
     * b) assume that the camera moves and we use the first transforms
     */
    VSTransform s_sum = null;
    for (int i = 0; i < td->conf.smoothing; i++){
      s_sum = add_transforms(&s_sum, i < trans->len ? &ts2[i]:&null);
    }
    mult_transform(&s_sum, 2); // choice b (comment out for choice a)

    for (int i = 0; i < trans->len; i++) {
      VSTransform* old = ((i - td->conf.smoothing - 1) < 0)
        ? &null : &ts2[(i - td->conf.smoothing - 1)];
      VSTransform* new = ((i + td->conf.smoothing) >= trans->len)
        ? &null : &ts2[(i + td->conf.smoothing)];
      s_sum = sub_transforms(&s_sum, old);
      s_sum = add_transforms(&s_sum, new);

      avg = mult_transform(&s_sum, 1.0/s);

      /* lowpass filter:
       * meaning high frequency must be transformed away
       */
      ts[i] = sub_transforms(&ts2[i], &avg);
      /* kill accumulating offset in the filtered signal*/
      avg2 = add_transforms_(mult_transform(&avg2, 1 - tau),
                             mult_transform(&ts[i], tau));
      ts[i] = sub_transforms(&ts[i], &avg2);

      if (td->conf.verbose & VS_DEBUG) {
        vs_log_msg(td->conf.modName,
                   "s_sum: %5lf %5lf %5lf, ts: %5lf, %5lf, %5lf\n",
                   s_sum.x, s_sum.y, s_sum.rotate,
                   ts[i].x, ts[i].y, ts[i].rotate);
        vs_log_msg(td->conf.modName,
                   "  avg: %5lf, %5lf, %5lf avg2: %5lf, %5lf, %5lf",
                   avg.x, avg.y, avg.rotate,
                   avg2.x, avg2.y, avg2.rotate);
      }
    }
    vs_free(ts2);
  }
  /* relative to absolute */
  if (td->conf.relative) {
    VSTransform t = ts[0];
    for (int i = 1; i < trans->len; i++) {
      ts[i] = add_transforms(&ts[i], &t);
      t = ts[i];
    }
  }
  return VS_OK;
}


/**
 * vsPreprocessTransforms: camera path optimization, relative to absolute conversion,
 *  and cropping of too large transforms.
 *
 * Parameters:
 *            td: transform private data structure
 *         trans: list of transformations (changed)
 * Return value:
 *     1 for success and 0 for failure
 * Preconditions:
 *     None
 * Side effects:
 *     td->trans will be modified
 */
//TODO [DEBUG] Apply zoom
int vsPreprocessTransforms(VSTransformData* td, VSTransformationsContainer* trans)
{
  // works inplace on trans
  if(cameraPathOptimization(td, trans)!=VS_OK) return VS_ERROR;
  VSTransform* ts = trans->ts;
  /*  invert? */
  if (td->conf.invert) {
    for (int i = 0; i < trans->len; i++) {
      ts[i] = mult_transform(&ts[i], -1);
    }
  }

  /* crop at maximal shift */
  if (td->conf.maxShift != -1)
    for (int i = 0; i < trans->len; i++) {
      ts[i].x     = VS_CLAMP(ts[i].x, -td->conf.maxShift, td->conf.maxShift);
      ts[i].y     = VS_CLAMP(ts[i].y, -td->conf.maxShift, td->conf.maxShift);
    }
  if (td->conf.maxAngle != - 1.0)
    for (int i = 0; i < trans->len; i++)
      ts[i].rotate = VS_CLAMP(ts[i].rotate, -td->conf.maxAngle, td->conf.maxAngle);

  if (trans->len > 1) {

	  if (td->conf.optZoom == 1){

	  /* Calc optimal zoom (1)
	   *  cheap algo is to only consider translations
	   *  uses cleaned max and min to eliminate 99% of transforms
	   */

		VSTransform min_t, max_t;
		cleanmaxmin_xy_transform(ts, trans->len, 1, &min_t, &max_t);  // 99% of all transformations
		// the zoom value only for x
		double zx = 2*VS_MAX(max_t.x,fabs(min_t.x))/td->fiSrc.width;
		// the zoom value only for y
		double zy = 2*VS_MAX(max_t.y,fabs(min_t.y))/td->fiSrc.height;
		td->conf.zoom += 100 * VS_MAX(zx,zy); // use maximum
		td->conf.zoom = VS_CLAMP(td->conf.zoom,-60,60);
		vs_log_info(td->conf.modName, "Final zoom: %lf\n", td->conf.zoom);

	  } else if (td->conf.optZoom == 2 && trans->len > 1){

	  /* Calc optimal zoom (2)
	   *  sliding average to zoom only as much as needed also using rotation angles
	   *  the baseline zoom is the mean required zoom + global zoom
	   *  in order to avoid too much zooming in and out
	   */
		double* zooms=(double*)vs_zalloc(sizeof(double)*trans->len);
		int w = td->fiSrc.width;
		int h = td->fiSrc.height;
		double req;
		double meanzoom;
		for (int i = 0; i < trans->len; i++) {
//		  VSTransform relativeTransform;
//		  if (i > 1) {
//			  relativeTransform = sub_transforms(&ts[i], &ts[i-1]);
//		  } else {
//			  relativeTransform = ts[i];
//		  }
//		  zooms[i] = transform_get_required_zoom(&relativeTransform, w, h);

		  zooms[i] = transform_get_required_zoom(&ts[i], w, h);


//		  fprintf(stderr, "Required zoom for [%i]: %f\n", i, zooms[i]);
		}
		meanzoom = mean(zooms, trans->len) + td->conf.zoom; // add global zoom
		// forward - propagation (to make the zooming smooth)
		req = meanzoom;
		for (int i = 0; i < trans->len; i++) {
		  req = VS_MAX(req, zooms[i]);
		  ts[i].zoom=VS_MAX(ts[i].zoom,req);
//		  fprintf(stderr, "Zoom out for [%i]: %f\n", i, ts[i].zoom);
		  req= VS_MAX(meanzoom, req - td->conf.zoomSpeed); // zoom-out each frame
//		  fprintf(stderr, "New req after [%i]: %f\n", i, req);
		}
		// backward - propagation
		req = meanzoom;
		for (int i = trans->len-1; i >= 0; i--) {
		  req = VS_MAX(req, zooms[i]);
		  ts[i].zoom=VS_MAX(ts[i].zoom,req);
//		  fprintf(stderr, "CP for [%i]: %f\n", i, ts[i].zoom);
		  req= VS_MAX(meanzoom, req - td->conf.zoomSpeed);
//		  fprintf(stderr, "New req(BP) after [%i]: %f\n", i, req);
		}
		vs_free(zooms);
	  }
  }

  if (td->conf.zoom != 0 && td->conf.optZoom != 2){ /* apply global zoom */
    for (int i = 0; i < trans->len; i++)
      ts[i].zoom += td->conf.zoom;
  }

  return VS_OK;
}


/**
 * vsLowPassTransforms: single step smoothing of transforms, using only the past.
 *  see also vsPreprocessTransforms. Here only relative transformations are
 *  considered (produced by motiondetection). Also cropping of too large transforms.
 *
 * Parameters:
 *            td: transform private data structure
 *           mem: memory for sliding average transformation
 *         trans: current transform (from previous to current frame)
 * Return value:
 *         new transformation for current frame
 * Preconditions:
 *     None
 */
VSTransform vsLowPassTransforms(VSTransformData* td, VSSlidingAvgTrans* mem,
                                const VSTransform* trans)
{

  if (!mem->initialized){
    // use the first transformation as the average camera movement
    mem->avg=*trans;
    mem->initialized=1;
    mem->zoomavg=0.0;
    mem->accum = null_transform();
    return mem->accum;
  }else{
    double s = 1.0/(td->conf.smoothing + 1);
    double tau = 1.0/(3.0 * (td->conf.smoothing + 1));
    if(td->conf.smoothing>0){
      // otherwise do the sliding window
      mem->avg = add_transforms_(mult_transform(&mem->avg, 1 - s),
                                 mult_transform(trans, s));
    }else{
      mem->avg = *trans;
    }

    /* lowpass filter:
     * meaning high frequency must be transformed away
     */
    VSTransform newtrans = sub_transforms(trans, &mem->avg);

    /* relative to absolute */
    if (td->conf.relative) {
      newtrans = add_transforms(&newtrans, &mem->accum);
      mem->accum = newtrans;
      if(td->conf.smoothing>0){
        // kill accumulating effects
        mem->accum = mult_transform(&mem->accum, 1.0 - tau);
      }
    }

    /* crop at maximal shift */
    if (td->conf.maxShift != -1){
      newtrans.x     = VS_CLAMP(newtrans.x, -td->conf.maxShift, td->conf.maxShift);
      newtrans.y     = VS_CLAMP(newtrans.y, -td->conf.maxShift, td->conf.maxShift);
    }
    if (td->conf.maxAngle != - 1.0)
      newtrans.rotate = VS_CLAMP(newtrans.rotate, -td->conf.maxAngle, td->conf.maxAngle);

    /* Calc sliding optimal zoom
     *  cheap algo is to only consider translations and to sliding avg
     */
    if (td->conf.optZoom != 0 && td->conf.smoothing > 0){
      // the zoom value only for x
      double zx = 2*newtrans.x/td->fiSrc.width;
      // the zoom value only for y
      double zy = 2*newtrans.y/td->fiSrc.height;
      double reqzoom = 100* VS_MAX(fabs(zx),fabs(zy)); // maximum is requried zoom
      mem->zoomavg = (mem->zoomavg*(1-s) + reqzoom*s);
      // since we only use past it is good to aniticipate
      //  and zoom a little in any case (so set td->zoom to 2 or so)
      newtrans.zoom = mem->zoomavg;
    }
    if (td->conf.zoom != 0){
      newtrans.zoom += td->conf.zoom;
    }
    return newtrans;
  }
}

/*
 * Local variables:
 *   c-file-style: "stroustrup"
 *   c-file-offsets: ((case-label . *) (statement-case-intro . *))
 *   indent-tabs-mode: nil
 *   c-basic-offset: 2 t
 * End:
 *
 * vim: expandtab shiftwidth=2:
 */
