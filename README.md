## Wavelet Tree Synthesis
<br />
A small project to test-drive the wavelet tree-learning & synthesis concepts from the following papers:

* [Dubnov, Shlomo, et al. "Synthesizing sound textures through wavelet tree learning." Computer Graphics and Applications, IEEE 22.4 (2002): 38-48.](http://www.cs.huji.ac.il/~danix/Dubnov-cga2002.pdf)

* [Misra, Ananya, Perry R. Cook, and Ge Wang. "Musical Tapestry: Re-composing Natural Sounds."](http://soundlab.cs.princeton.edu/publications/taps_icmc2006.pdf)


The treesynth and daubachies source was excised from the source code distribution of [TAPESTREA](http://taps.cs.princeton.edu/) and modified to compile and run. The UI dependencies were also ripped out, so the resultant project builds a command line app that does wavelet tree synthesis, given an input wave file.
<br /> 

### Takeaways from these experiments

- Implementation per the paper resulted in a “chattering” or “stuttering” effect from portions of the signal being repeated. 
- The signal decomposition did not appear to respect a natural segmentation of sonic events in a way that their recombination (during synthesis) was smooth and natural. 
- **Hoskinson, Reynald. Manipulation and resynthesis of environmental sounds with natural wavelet grains. Diss. The University of British Columbia, 2002.** state that “these convolution artifacts arise because switching coefficients of the wavelet transform has unpredictable results. Unless the changes are on the dyadic boundaries, it is really changing the timbre of the input sound rather than switching the sound events. These changes cannot be easily predicted; they have to do with the choices of wavelet filter, the filter length, and the position of the coefficients. The convolution involved in reconstruction makes this process virtually impossible do without introducing unwanted artifacts.”
- Other issues include the complexity of successfully (sans-artifiacts) extending the length of the signal to arbitrary length or even a power-of-two.
- [Wavelet packets](https://en.wikipedia.org/wiki/Wavelet_packet_decomposition) might be an alternative to the daub10 based multi-resolution wavelet tree, but the literature suggests that wavelet packets are not guaranteed to represent, entire re-arrangeable events.

### Credit

**TAPESTREA: Techniques And Paradigms for Expressive Synthesis, Transformation, and Rendering of Environmental Audio Engine and User Interface**

Copyright ©  2006 Ananya Misra, Perry R. Cook, and Ge Wang.

- [http://taps.cs.princeton.edu](http://taps.cs.princeton.edu)
- [http://soundlab.cs.princeton.edu](http://soundlab.cs.princeton.edu)


### License
Everything GPLv2, per the original TAPESTRA 0.1.0.6 release
