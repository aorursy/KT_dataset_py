from IPython.core.display import SVG

from IPython.display import Image
# Reference: https://corporatefinanceinstitute.com/resources/careers/soft-skills/t-shaped-skills/

Image(filename='../input/web-app-images/t-shaped-skills-1.png', width = 300)
SVG(filename='../input/web-app-images/pocoo_flask-official.svg')
### app.py

# from flask import Flask

# app = Flask(__name__)

# @app.route("/")

# def hello():

#     return "Hello World!"

# if __name__ == "__main__":

#     app.run(debug=True)
### model.py

# import pickle

# import numpy as np

# from sklearn.linear_model import LinearRegression

# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

# y = np.dot(X, np.array([1, 2])) + 3

# if __name__ == "__main__":

#     reg = LinearRegression().fit(X, y)

#     PIK = "pickle.dat"

#         with open(PIK, 'wb') as f:

#             pickle.dump(reg, f)
### app.py

# import pickle

# import model // imports model.py

# if __name__ == "__main__":

#     PIK = 'pickle.dat'

#     with open(PIK, "rb") as f:

#         loaded_agents = pickle.load(f)
### app.py

# if __name__ == "__main__":

#     port = int(os.environ.get('PORT', 5000))

#     app.run(host='0.0.0.0', port=port)
SVG(filename='../input/web-app-images/docker-official.svg')
Image(filename='../input/web-app-images/docker_vm.png', width=700)
# FROM python:3



# WORKDIR /usr/src/app



# COPY requirements.txt ./

# RUN pip install --no-cache-dir -r requirements.txt



# COPY . .



# CMD [ "python", "./app.py" ]
SVG(filename='../input/web-app-images/heroku-ar21.svg')