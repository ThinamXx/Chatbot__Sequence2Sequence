# **Chatbot using Sequence to Sequence Neural Networks**

**Sequence to Sequence Networks**
- Sequence to Sequence Neural Networks can be built with a modular and reusable Encoder and Decoder Architecture. The Encoder Model generates a Thought Vector which is a Dense and fixed Dimension Vector representation of the Data. The Decoder Model use Thought Vectors to generate Output Sequences including the replies of the Chatbot. Due to Thought Vector Representation the Input and Output, Sequence lengths don't have to match while preparing the Dataset for Training the Model. Thought Vectors can only hold a limited amount of Information. If the Implementation of Thought Vector is made using complex concepts then the Attention Mechanisms can help to encode the Important Thought Vector selectively.

**Libraries and Dependencies**
- I have listed all the necessary Libraries and Dependencies required for this Project here:

```javascript
import numpy as np                                                         
from nlpia.loaders import get_data 
import os
from random import shuffle                                                 
from IPython.display import display

from keras.models import Model
from keras.layers import Input, LSTM, Dense
```

**Getting the Dataset**
- I have used Google Colab for this Project so the process of downloading and reading the Data might be different in other platforms. I have used [**Cornell Movie Dialog Dataset**](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) for this Project. Using the entire [Cornell Movie Dialog Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) can be computationally intensive because a few sequences have more than 2000 tokens. I have used NLPIA Package to load the **Cornell Movie Dialog Dataset**.

**Processing the Dataset**
- I have presented the simple Implementation of processing the Text Corpus and the process of building the Character Dictionary to make the Text Data ready to train the Sequence to Sequence Chatbot here in the Snapshot. I will convert each characters of the Input and Target Texts into one hot vectors that represent each characters. In order to generate one hot vectors I will generate token dictionaries where every character is mapped to an index. I will also generate the reverse dictionaries which will be used to convert generated index into characters.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2032.PNG)

### **Sequence to Sequence Chatbot**
- Sequence to Sequence Neural Networks can be built with a modular and reusable Encoder and Decoder Architecture. The Encoder Model generates a Thought Vector which is a Dense and fixed Dimension Vector representation of the Data. The Decoder Model use Thought Vectors to generate Output Sequences including the replies of the Chatbot. Due to Thought Vector Representation the Input and Output, Sequence lengths don't have to match while preparing the Dataset for Training the Model. Thought Vectors can only hold a limited amount of Information. If the Implementation of Thought Vector is made using complex concepts then the Attention Mechanisms can help to encode the Important Thought Vector selectively. 
- I have presented the Implementation of Thought Encoder and Thought Decoder using Keras Functional API here in the Snapshot. I have also presented the techniques for Training the Model and Generating the Response Sequences here:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2033.PNG)

**Saving the Model**

```javascript
model_structure = model.to_json()
with open("sequence_model.json", "w") as json_file:
  json_file.write(model_structure)
model.save_weights("sequence_model.h5")
```
