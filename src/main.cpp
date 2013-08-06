//
//  main.cpp
//  Treesynth
//
//  Created by iroro orife on 8/1/12.
//
//  I excised & modified a few files from TAPESTRA (see below) to
//  test-drive wavelet tree-learning & synthesis concepts from
//  the following papers:
//
//  Dubnov, Shlomo, et al. "Synthesizing sound textures through
//  wavelet tree learning." Computer Graphics and Applications,
//  IEEE 22.4 (2002): 38-48.
//
//  Misra, Ananya, Perry R. Cook, and Ge Wang. "Musical Tapestry:
//  Re-composing Natural Sounds."
//
//  Everything GPLv2, per the original TAPESTRA 0.1.0.6 release
//
//
//
// TAPESTREA: Techniques And Paradigms for Expressive Synthesis,
// Transformation, and Rendering of Environmental Audio
// Engine and User Interface
//
// Copyright (c) 2006 Ananya Misra, Perry R. Cook, and Ge Wang.
// http://taps.cs.princeton.edu/
// http://soundlab.cs.princeton.edu/
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// U.S.A.



#include <iostream>
#include "taps_treesynth.h"

int main(int argc, const char * argv[]){
	// make new Treesynth and TreesynthIO objects for template
	Treesynth * ts = new Treesynth();
	TreesynthIO * tsio = new TreesynthIO();

	char in_path[1024] = "/Users/iorif/Desktop/goodbye_hollis_etta_SHORT.wav";
	char out_path[1024] = "/Users/iorif/Desktop/treesyn_OUT.wav";

	strcpy(tsio->ifilename, in_path);
	strcpy(tsio->ofilename, out_path);

	ts->tree = new Tree();                          // new tree for tree synth
	ts->tree->initialize(lg(CUTOFF));             // initialize tree to log base 2 of 36 == 8
	ts->initialize();                               // init tree synth
	ts->resetTreeLevels(13);                      // log2 (8192) == 13, buffersize is 8192

	// treesynth knobs  -- Need better explanations for these params
	ts->kfactor = 1.0;     // ui_elements[SL_K]->fvalue();             // Determines npredecessors
	ts->randflip = true;    // ui_elements[FL_RANDFLIP]->slide >.5;     // Whether first 2 coefficients are copied in order or randomly
	ts->percentage = 1.0;   // ui_elements[SL_PERCENTAGE]->fvalue();    // Percentage of nodes to be considered when learning new tree
	ts->ancfirst = true;    // ui_elements[FL_ANCFIRST]->slide > .5;    // Whether learning is first done on ancestors or predecessors
	ts->startlevel = 1;     // ui_elements[SL_STARTLEVEL]->ivalue();
	ts->stoplevel = 9;      // ui_elements[SL_STOPLEVEL]->ivalue();     // Changed later in the program


	int samples, write = 1;
	while(1){
		// Zero the trees
		ts->resetTrees();

		samples = tsio->ReadSoundFile(tsio->ifilename, ts->tree->values(), ts->tree->getSize() /* 1 << requested_levels */);
		if(samples <= 0){
			printf("read %i samples from the file %s \n\n", samples, tsio->ifilename);
			break;
		}
		printf("Read %d samples (%d); requested: %d\n", samples, lg(samples), ts->tree->getSize() /* 1 << requested_levels */);
		// Change levels accordingly

		ts->resetTreeLevels(lg(samples));
		TS_UINT total_levels = ts->tree->getLevels();
		printf("total_levels: %d \n", total_levels);

		if(ts->setup()){
			ts->synth();
			// keep trying to write to file/buffer, until written or told to shut up

			while(!(write = tsio->WriteSoundFile(tsio->ofilename, ts->outputSignal(), ts->tree->getSize()))){
				//res->mutex.release();
				//BB_log( BB_LOG_FINE, "ts_go : released mutex" );

				usleep(10000);

				//res->mutex.acquire();
				//BB_log( BB_LOG_FINE, "ts_go : acquired mutex" );
			}
		}
	}

	return 0;
}