# ML-Projects
## Text based Projects: 
1. LSTM movie preview prediction : Keras, tokenizers, stemming, Embedding, LSTM
   
    download dataset on colab using wget command
   
    dataset contains data scraped from web using some web scraping commands so it has html tags and many other unwanted features with it which needs to be removed.
   
    Remove stop words: better make own stopword list for sentiment analysis types task since there can be some stopwords which might be useful for analysing sentiments.
   
    tokenization : convert sentence to words and remove special characters which are of no use.
   
    stemming: converting words back to their original form since they might have been changed to their other form like work -> working -> worked etc
   
    converting string to their numeric notation by mapping each unique word of input data to some nemeric data which is further going to be changed into a vector of size 50 according to their sentimental values.
   
    Reducing unwanted words like words which are occuring very less frequently and are having very small length
   
    converting each sentence of different length to same length vector we need to analyze what their length should be according to input data variation.
   
    creating model using Embedding + LSTM + Dense layers
   
    tranforming testing dataset same way as we did with training dataset.
   
    predicting over trained model and creating csv file of testing data output
   
    submitting to online judge and checking accracy

   
3. torch_torchText : torch, RNN
   
   This is character based text generation model to generate new names based on previous dataset of names
   
   This was to learn creating torch layers from scrach and converting text dataset to such form that can be feeded to deep models specially LSTM in this case
   
   We converted each name to a stream of character which is then converted to one-hot vector
   
   These one-hot vectors are then fed to LSTM to make prediction
   
   We also utiized Embedding layers to convert our one-hot-vectors to sentimental vector represenation

   
## Image based Projects 
3. MNIST_GANN : keras, matplotlib, CNN
   
   Generative Adverserial Networks created to generate dataset similar to MNIST dataset which have good results can be varified visually
   
   It involves some image preprocessing using matplotlib to read images and printing them in desired format

   GANN are having 2 models generator and discriminator
   
   generator : it's work is to fool discriminator generate images such that discriminator thinks these are real images.
   
   discriminator : it's work is to distinguish between real (dataset images), fake (generator generated images) images.
   
   During training first discriminator is trained by making generator off and then generator trained by turning off discriminator model

   Learnt few things to be taken care of while creating such generative models.

   
5. conditional_GANN_Cifar10 : keras, OpenCV
   
   conditional GaNN are similar to GANN but they are trained not to generate random data instead based on some input they adds features to images
   
   A good example could be adding specs data + face data as input this is supposed to generate face with specs on it.
   
   Similarly here we are taking simple input 1..10 that signifies which class data to be generated means when input is 1 it should generated data only of types class 1.
   
   This conditioning of GANN is learnt in this model
7. CardioEjectionFractionPrediction : VideoDataset, Keras, OpenCV
   
   This project works on video dataset of Heart ultrasound.
   
   Our goal was to optimize prediction as it's working on video dataset that will require lot of memory and computation for make prediction.
   
   So to optimize over that we used auto-encoder and fed 50 frames of video to our model.
   
   auto-encoder then convert these 50 frames to single frame this gives a optimization in terms of space and memory.
   
   After then using this optimized vector to make a prediction model that was able to give considerably less amount of loss making this a good model for this purpose. 

