
# Vision Transformers

The notebook presented in this repository contains a walk through of the Vision Transformer model with illustrations. 

The notebook contains a step-by-step implementation of the paper '**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**'.

For the interactive version, visit [this Colab Notebook](https://colab.research.google.com/drive/1eQS3NZ8gPhTGd98o88rKZ0nYZJMg7Ujq).

![Vision Transformer](https://cdn-images-1.medium.com/max/3492/1*zjObybet7mzE18rnhYtz4A.png)<p align='center'>*Vision Transformer*</p>

## High — Level Look

![**The Vision Transformer model takes an image and outputs the class of the image**](https://cdn-images-1.medium.com/max/2944/1*Tmd2SCYOHtG53jCin4MkOg.png)<p align='center'>*The Vision Transformer model takes an image and outputs the class of the image*</p>



The Vision Transformer model consists of the following steps:

1. Split an image into fixed-size patches

1. Linearly embed each of the patches

1. Prepend [class] token to embedded patches

1. Add positional information to embedded patches

1. Feed the resulting sequence of vectors to a stack of standard Transformer Encoders

1. Extract the [class] token section of the Transformer Encoder output

1. Feed the [class] token vector into the classification head to get the output


## Image Patches

We start by splitting an image into fixed-size patches.

![128 x 128 resolution image to 64 number of 16 x 16 resolution patches](https://cdn-images-1.medium.com/max/2386/1*oFOsZa3Al81f4nXhkJ3dkw.png)<p align='center'>*128 x 128 resolution image to 64 number of 16 x 16 resolution patches*</p>

We order the sequence of patches from top left to bottom right. After ordering, we flatten these patches.

![We get a linear sequence with flattening](https://cdn-images-1.medium.com/max/3014/1*zsqLltq9swy-YQJfQlB9Kw.png)<p align='center'>*We get a linear sequence with flattening*</p>

## Embedding Patches

To generalize these steps for any image, we take an image with a resolution of height and width, and split it into a number of patches with a specified resolution that we call **patch size**, and flatten these patches.

![](https://cdn-images-1.medium.com/max/3116/1*oGvOTvDWtN-_U-lbVtGTOA.png)

We multiply the flattened patches with a weight matrix to obtain the **embedding dimensionality** that we want.

![Multiply with a trainable linear projection to get embedding dimensionality, D = 768 in the paper](https://cdn-images-1.medium.com/max/2858/1*rcUMaZhSci593f5FKZgZbA.png)<p align='center'>*Multiply with a trainable linear projection to get embedding dimensionality, D = 768 in the paper*</p>

Now, as in the standard Transformer, we have a sequence of linear embeddings with a defined dimensionality.

## [Class] Token

We continue by prepending a [class] token to the patch embeddings. The [class] token is a randomly initialized learnable parameter that can be considered as a placeholder for the classification task.

The token gathers information from the other patches while flowing through the Transformer Encoders and the components inside the encoders. To be not biased toward any of the patches, we use the [class] token as a **decoder**, we apply the classification layer on the corresponding [class] token part of the output.

![](https://cdn-images-1.medium.com/max/2744/1*x9kAc8a9PU1pXOS4g5B38A.png)

## **Positional Information**

To describe the location of an entity in a sequence, we use **Positional Encoding** so that each position is assigned to a unique representation.

We add the positional embeddings with the intention of injecting location knowledge to the patch embedding vectors. Upon training, the position embeddings learn to determine the position of given image in a sequence of images.

![Position embeddings are learnable parameters](https://cdn-images-1.medium.com/max/2990/1*FbtExz-mw-XbGyMSVeATCw.png)<p align='center'>*Position embeddings are learnable parameters*</p>

With this final step, we obtain the embeddings that have all the information that we want to pass on. We feed this resulting matrix into a stack of Transformer Encoders.

![](https://cdn-images-1.medium.com/max/3642/1*_anaSSdAE4w7s2DYMo_mnA.png)

Let’s dive into the structure of the Transformer Encoder.

## Transformer Encoder

The Transformer Encoder is composed of two main layers: **Multi-Head Self-Attention** and **Multi-Layer Perceptron**. Before passing patch embeddings through these two layers, we apply **Layer Normalization** and right after passing embeddings through both layers, we apply **Residual Connection**.

![Transformer Encoder](https://cdn-images-1.medium.com/max/3710/1*SKsN3HlCrDNxLkLOYlV55A.png)<p align='center'>*Transformer Encoder*</p>

Let’s look into the structure of the Multi-Head Self-Attention layer. The **Multi-Head Self-Attention** layer is composed by a number of **Self-Attention** heads running in parallel.

## Self-Attention

The **Self-Attention mechanism** is a key component of the Transformer architecture, which is used to capture contextual information in the input data. The self-attention mechanism allows a Vision Transformer model to attend to different regions of the input data, based on their relevance to the task at hand. The Self-Attention mechanism uses key, query and value concept for this purpose.

The key/value/query concept is analogous to retrieval systems. For example, when we search for videos on Youtube, the search engine will map our **query** (text in the search bar) against a set of **keys** (video title, description, etc.) associated with candidate videos in their database, then present us the best matched videos (**values**). The dot product can be considered as defining some similarity between the text in search bar (query) and titles in the database (key).

To calculate self-attention scores, we first multiply our input sequence with a single weight matrix (that is actually a uniform of three weight matrices). Upon multiplying, **for each patch, we get a query vector, a key vector, and a value vector**.

![Key, Query and Value Matrices](https://cdn-images-1.medium.com/max/3242/1*xhv57fEpmgnTAoFWsW3_Aw.png)<p align='center'>*Key, Query and Value Matrices*</p>

We compute the dot products of the query with all keys, divide each by √ key dimensionality, and apply a softmax function to obtain the weights on the values. Softmax function normalizes the scores so they are all positive and add up to 1.

We then multiply the attention weights with values to get the Self-Attention output.

![Weighted Values](https://cdn-images-1.medium.com/max/3186/1*naF0F0IPbt4CdOQyhNMrcw.png)<p align='center'>*Weighted Values*</p>

**Note that** these calculations are done for each head separately. The weight matrices are also initialized and trained for each head separately.

## Multi-Head Self-Attention

In order to have multiple **representation subspaces** and **to attend** to multiple parts on an input, we repeat these calculations for each head and concatenate these Self-Attention outputs. After that, we multiply the result with a weight matrix to reduce the dimensionality of the output that grows in size with concatenation.

![Obtaining results for each Attention Head, the number of heads is 12 in the paper](https://cdn-images-1.medium.com/max/3598/1*r4Y9Y3YefWcsUyHpWTvNxw.png)<p align='center'>*Obtaining results for each Attention Head, the number of heads is 12 in the paper*</p>

![Getting Multi-head Self-Attention score](https://cdn-images-1.medium.com/max/3246/1*Pql8Xq_LPruhUHhQoIOhsg.png)<p align='center'>*Getting Multi-head Self-Attention score*</p>

That is pretty much all there is to Multi-Head Self-Attention calculations.

Now, we will step into the other layer that is in the Transformer Encoder which is the Multi-Layer Perceptron.

## Multi-Layer Perceptron

Multi-Layer Perceptron is composed of two hidden layers with a **Gaussian Error Linear Unit** activation function in-between the hidden layers.

GELU, has a smoother, more continuous shape than the ReLU function, which can make it more effective at learning complex patterns in the data.

![The hidden layers’ dimensionality is 3072 in the paper](https://cdn-images-1.medium.com/max/2360/1*rU4YLEDOS4M6ErqcL-MtMw.png)<p align='center'>*The hidden layers’ dimensionality is 3072 in the paper*</p>

Thus, we are done with the components of the Transformer Encoder.

We are left with one final component that makes up the Vision Transformer model which is the Classification Head.

If we continue from where we left off, after passing embedded patches through the Transformer Encoder stack, we achieved a Transformer Encoder output.

## Classification Head

We extract the [class] token part of the Transformer Encoder output and pass it through a classification head to get the class output.

![[Class] Token of Output](https://cdn-images-1.medium.com/max/2394/1*RdCPl1XZGD72SXV_oR0Gcw.png)<p align='center'>*[Class] Token of Output*</p>

![Classification](https://cdn-images-1.medium.com/max/2134/1*UuiIdiO3yQz3aEdNNkiKxg.png)<p align='center'>*Classification*</p>

Finally, by applying a Softmax function to the output layer, we obtain the probabilities of the class outputs.


## Resources

1. [Transformers For Image Recognition At Scale — Paper](https://arxiv.org/pdf/2010.11929.pdf)

1. [Illustrated Transformer — Blog Post](https://jalammar.github.io/illustrated-transformer/)

1. [Attention is All You Need — Paper](https://arxiv.org/pdf/1706.03762.pdf)

1. [Key, Value and Query Concept](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms)

1. [Vision Transformer in Image Recognition — Blog Post](https://viso.ai/deep-learning/vision-transformer-vit/)

1. [[Class] Token in ViT and BERT](https://datascience.stackexchange.com/questions/90649/class-token-in-vit-and-bert)

### Additional Resources

1. [Layer Normalization — Tutorial Video](https://www.youtube.com/watch?v=2V3Uduw1zwQ)

1. [Transformers —Tutorial ](https://www.youtube.com/watch?v=_UVfwBqcnbM&t=1s)[Video](https://www.youtube.com/watch?v=_UVfwBqcnbM&t=1s)

