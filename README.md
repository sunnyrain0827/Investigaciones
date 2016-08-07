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


### Seq2Seq papers, code and DL toolkit summaries

| Summary | Code | Papers | Framework |
| ----- | ---- | ---- | ---- |
| Seq2Seq | https://github.com/sq6ra/encoder-decoder-model-with-sequence-learning-with-RNN-LSTM- | [sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215) | Theano |
| Seq2Seq | https://github.com/farizrahman4u/seq2seq, https://github.com/nicolas-ivanov/debug_seq2seq | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | Keras |
| Skip Thought Seq2Seq  | [https://github.com/zuiwufenghua/Sequence-To-Sequence-Generation-Skip-Thoughts-](https://github.com/zuiwufenghua/Sequence-To-Sequence-Generation-Skip-Thoughts-) | [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726) | Keras |
| Seq2Seq with Attention  | [https://github.com/harvardnlp/seq2seq-attn](https://github.com/harvardnlp/seq2seq-attn) | [Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf), [Character-Aware Neural Language Models](http://arxiv.org/abs/1508.06615) | Torch |
| Variational Seq2Seq  | [https://github.com/cheng6076/Variational-LSTM-Autoencoder](https://github.com/cheng6076/Variational-LSTM-Autoencoder) | [Variational Recurrent Auto-Encoders](http://arxiv.org/abs/1412.6581),  [Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)  | Torch |
| Variational Recurrent Autoencoder (VRAE)| [https://github.com/y0ast/Variational-Recurrent-Autoencoder](https://github.com/y0ast/Variational-Recurrent-Autoencoder) | [Variational Recurrent Auto-Encoders](http://arxiv.org/abs/1412.6581)  | Theano |
| Variational Recurrent Autoencoder (VRAE) | [https://github.com/RyotaKatoh/chainer-Variational-Recurrent-Autoencoder](https://github.com/RyotaKatoh/chainer-Variational-Recurrent-Autoencoder) | [Variational Recurrent Auto-Encoders](http://arxiv.org/abs/1412.6581)  | Chainer |
| LSTM VRAE | [https://github.com/jayhack/LSTMVRAE](https://github.com/jayhack/LSTMVRAE) | [Variational Recurrent Auto-Encoders](http://arxiv.org/abs/1412.6581) | Chainer |
| Variational Recurrent Neural Networks (VRNN) | [https://github.com/jych/nips2015_vrnn](https://github.com/jych/nips2015_vrnn) | [A Recurrent Latent Variable Model for Sequential Data](http://arxiv.org/abs/1412.6581)  | Theano |




### Sequence learning variants

| Summary | Code | Papers | Framework |
| ----- | ---- | ---- | ---- |
| Generative Handwriting using LSTM Mixture Density Network  | [https://github.com/hardmaru/write-rnn-tensorflow](https://github.com/hardmaru/write-rnn-tensorflow)| [Generating Sequences With Recurrent Neural Networks](http://arxiv.org/abs/1506.02216)  | TensorFlow |
| DRAW | [https://github.com/ericjang/draw](https://github.com/ericjang/draw) | [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/abs/1502.04623)  | TensorFlow |
| Generating sound with RNNs  | [https://github.com/johnglover/sound-rnn](https://github.com/johnglover/sound-rnn)| [Generating Sequences With Recurrent Neural Networks](http://arxiv.org/abs/1308.0850)  | Torch |
| VAE-GAN-DRAW | [https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW](https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW) | [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) & [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623) | TensorFlow |
| Tree RNN | [https://github.com/ofirnachum/tree_rnn](https://github.com/ofirnachum/tree_rnn) | [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)  | Theano |
| Sequence GAN | [https://github.com/ofirnachum/sequence_gan](https://github.com/ofirnachum/sequence_gan)| N/A | TensorFlow |
| Generative Recurrent Adversarial Network (GRAN)  | [https://github.com/jiwoongim/GRAN](https://github.com/jiwoongim/GRAN)| [Generating images with recurrent adversarial networks](http://arxiv.org/abs/1602.05110)  | Theano |
