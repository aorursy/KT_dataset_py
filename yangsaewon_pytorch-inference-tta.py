import numpy as np

import pandas as pd

from pathlib import Path
df = pd.read_csv("../input/test.csv")

df.head()
weights_path = Path('../input/../..')

weight_list = os.listdir(weights_path)
batch_size = 1

tta = 3

test_dataset = TestDataset(...)

test_loader = DataLoader(...)

total_num_models = len(weight_list)*tta 



model = ...

model.cuda()





all_prediction = np.zeros((len(test_dataset), num_classes))



for i, weight in enumerate(weight_list):

    print("fold {} prediction starts".format(i+1))

    

    for _ in range(tta):

        print("tta {}".format(_+1))



        model.load_state_dict(torch.load(weights_path / weight))



        model.eval()

        

        prediction = np.zeros((len(test_dataset), num_classes)) # num_classes=196

        with torch.no_grad():

            for i, images in enumerate(test_loader):

                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()

                all_prediction = all_prediction + prediction

    

all_prediction /= total_num_models

result = np.argmax(all_prediction, axis=1)



submission = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/sample_submission.csv')

submission["class"] = result

submission.to_csv("submission.csv", index=False)

submission.head()