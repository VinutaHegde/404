# Album Narration
## The problem we are trying to solve
## Dataset
We use VIST dataset for our training and testing. Picking the grains from a dataset was very interesting journey for us. There were few surprises on the way. Here is something users of VSIT dataset should know about before using it:
- Not all the Albums have stories associated with them
- Similarly, not all the available stories have albums associated with them
- Most of the albums have multiple stories associated with them.

Once we filtered albums with images as well as associated stories, we started preprocessing out data. <br>
Structure of the data looks like :<br>
> {Data:<br>
>> {[album1:{[image1, image2, image3, image4, image5],[caption1, caption2, caption3, caption4, caption5]}]<br>
 >>      [[album2:{[image1, image2, image3, image4, image5],[caption1, caption2, caption3, caption4, caption5]}]<br>
 >>      .<br>
 >>      .<br>
 >>      [[albumn:{[image1, image2, image3, image4, image5],[caption1, caption2, caption3, caption4, caption5]}]}}<br>

### Data preprocessing
We used pretrained __Xception model__ from Keras to extract the features from the images. To represent the captions we went with __GLOVE__ embedding. Each image is represented as __[2048 dimension]__ vector and each caption is represented as __[300*max_sentences_length__] 

#### Training pair for Base Model1:


#### Training pair for Base Model2:



#### Training pair for final model:


##  Glossary
#### Album 
Album is a group of "n" images, Which might or might not have high visual corelation. (However a creative mind can generate a story by linking the dots) 
#### Caption
Caption is a group of words describing a given image.
#### Story 
Story is a group of captions describing an album (or sequence of images).
#### Image Embedding
#### Word Embeddings
#### VIST
