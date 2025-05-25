#  Histology Semantic Segmentation using Transfer Learning from Vision Transformer
by: Nitzan Haim, Shani Gindi

We performed semantic segmentation on pathological histology images using the Segmenter model (https://github.com/rstrudel/segmenter) with transfer
learning as well as training it from scratch. Using a pre-trained model as a starting point and continuing the training
on the histology data has shown the best results, compared to training from scratch and to fine-tuning only the
decoder.

**Why transfer learning on histology images?** Automating semantic segmentation is beneficial by itself, for example in research studies including large
amounts of specimens, as the manual examination of tissue is subjective and not scalable. We also trained
the model specifically on breast cancer data. Finetuning this model for a short time to other specific tissue types
showed great improvement in the model performance. For clinical and logistical reasons, it is often the case that
there is only a specific kind of data available and the need arises to generalize a pre-trained model on other types of
tissues which are poor in data.

------

This project was done as a final assignment for the Deep Learning for Computer Vision course at Weizmann.

To read the full report, including a full description of the input data, implementation, model evaluation and performance analysis, please visit: https://drive.google.com/file/d/1TQZvuffWgp-Z6ozp0FL5BOfVR2y6YmPz/view
