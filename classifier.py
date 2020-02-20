from keras.models import model_from_json
import cv2
import numpy as np

# Step 2: Load the Model from Json File
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Step 3: Load the weights
loaded_model.load_weights("./model.h5")
print("Loaded model from disk")

# Step 4: Compile the model
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Step 5: load the image you want to test
image = cv2.imread(r'.\test_set\Bowler\image08.png')
image = cv2.resize(image, (50,50))
image = image.reshape(1, 50, 50, 3)


# Step 6: Predict to which class your input image has been classified
result = loaded_model.predict_classes(image)
if(result[0][0] == 1):
    print("I guess this must be a Bowler!")
else:
    print("I guess this must be a Batsman!")
