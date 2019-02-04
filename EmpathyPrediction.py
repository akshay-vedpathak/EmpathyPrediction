import util
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from baseline import BaselinePredictor
from sklearn.svm import SVC

data = util.load_data()

preprocessed_data = util.preprocess_data(data)

X,Y = util.splitFeaturesAndLabel(preprocessed_data,'Empathy')

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42)

baseline_predictor = BaselinePredictor()

baseline_preds = util.trainAndPredict(X_train,Y_train,baseline_predictor,X_test)

print("Baseline accuracy and classification report")

util.printAccuracyAndClassficationReport(baseline_preds,Y_test,classes=['1','2','3','4','5'])

X_train,X_test = util.getBestFeatures(X_train,Y_train,X_test)

model = SVC(kernel='rbf')

params = {'C':[i for i in range(1,11)],
          'gamma':[0.1,0.01,0.001,0.0001,0.00099,0.000099]}

hyper_parameters = util.getBestHyperParameters(model,X_train,Y_train,params)

c = hyper_parameters.get('C')

gamma = hyper_parameters.get('gamma')

model = SVC(kernel='rbf',C=c,gamma=gamma)

preds = util.trainAndPredict(X_train,Y_train,model,X_test)

util.printAccuracyAndClassficationReport(preds,Y_test,classes=['1','2','3','4','5'])