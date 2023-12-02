# stabilityai/sdxl-turbo

This is an implementation of the [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="21 years old girl,short cut,beauty,dusk,Ghibli style illustration"

## Example:

"21 years old girl,short cut,beauty,dusk,Ghibli style illustration"

![21 years old girl,short cut,beauty,dusk,Ghibli style illustration](out-0.png)

# Note:
The model is intended for research purposes only. Possible research areas and tasks include

 - Research on generative models.
 - Research on real-time applications of generative models.
 - Research on the impact of real-time generative models.
 - Safe deployment of models which have the potential to generate harmful content.
 - Probing and understanding the limitations and biases of generative models.
 - Generation of artworks and use in design and other artistic processes.
 - Applications in educational or creative tools.