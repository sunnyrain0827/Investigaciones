# Investigaciones

* ~~Wavelet-Tree-Synth for texture synthesis~~
* ~~Time-signature detection using multi-res, similarity matrices~~
* Structural segmentation using MSAF: <https://github.com/ruohoruotsi/msaf>
* nnet - experiments in Keras/Theano/Tensorflow/Chainer:  
	* Biaxial RNN: <https://github.com/ruohoruotsi/biaxial-rnn-music-composition>
	* Sequence-to-sequence (LSTM-based) autoencoders:
		* alternative to ordinal linear discriminant analysis (OLDA)? Generalized automatic feature representation from a audio-similarity matrix
		* generalized audio/midi sequence generation 
	* Variational autoencoders to explore various structures of the learned manifold (latent space)
	* Variational recurrent autoencoders that map sequences of different lengths to a single length (re: olda, audio2vec, etc)
	* Info-GAN, Drum, Alex Graves vocoder
	* Finetuning VGG16 model


#### Seq2Seq paper, code and DL toolkit summaries


| Summary  | Code | Papers | Framework  |
| ------------- | ------------- | ---- |
|Seq2Seq | [https://github.com/sq6ra/encoder-decoder-model-with-sequence-learning-with-RNN-LSTM-](https://github.com/sq6ra/encoder-decoder-model-with-sequence-learning-with-RNN-LSTM-) | [sequence to sequence learning with neural networks](http://arxiv.org/pdf/1409.3215v3.pdf ) | Theano |
|Seq2Seq | [https://github.com/farizrahman4u/seq2seq](https://github.com/farizrahman4u/seq2seq) [https://github.com/nicolas-ivanov/debug_seq2seq](https://github.com/nicolas-ivanov/debug_seq2seq)| [sequence to sequence learning with neural networks](http://arxiv.org/pdf/1409.3215v3.pdf) | Keras |
| Skip Thought Seq2Seq  | [https://github.com/zuiwufenghua/Sequence-To-Sequence-Generation-Skip-Thoughts-](https://github.com/zuiwufenghua/Sequence-To-Sequence-Generation-Skip-Thoughts-) | [Skip-Thought Vectors](http://arxiv.org/pdf/1506.06726v1.pdf) | Keras| 
| Variational Seq2Seq  | [https://github.com/cheng6076/Variational-LSTM-Autoencoder](https://github.com/cheng6076/Variational-LSTM-Autoencoder) | [Variational Recurrent Auto-Encoders](http://arxiv.org/abs/1412.6581) -- [Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)  | Torch |
| Variational Recurrent Autoencoder (VRAE)| [https://github.com/RyotaKatoh/chainer-Variational-Recurrent-Autoencoder](https://github.com/RyotaKatoh/chainer-Variational-Recurrent-Autoencoder) | [Variational Recurrent Auto-Encoders](http://arxiv.org/abs/1412.6581)  | Chainer |
| Variational Recurrent Autoencoder (VRAE)| [https://github.com/y0ast/Variational-Recurrent-Autoencoder](https://github.com/y0ast/Variational-Recurrent-Autoencoder) | [Variational Recurrent Auto-Encoders](http://arxiv.org/abs/1412.6581)  | Theano |
| Tree RNN | [https://github.com/ofirnachum/tree_rnn](https://github.com/ofirnachum/tree_rnn) | [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)  | Theano |
| Sequence GAN | [https://github.com/ofirnachum/sequence_gan](https://github.com/ofirnachum/sequence_gan)| N/A | TensorFlow |
| Generative Recurrent Adversarial Network (GRAN)  | [https://github.com/jiwoongim/GRAN](https://github.com/jiwoongim/GRAN)| [Generating images with recurrent adversarial networks](https://arxiv.org/pdf/1602.05110.pdf)  | Theano |



