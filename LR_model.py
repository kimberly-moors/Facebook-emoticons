  #Logistic Regression
	def LR_model(train_f, test_f, y1, y2):
		""" This function creates the machine learning models for the Logistic Regression method.
		The function optimizes the model through Gridsearch. Then, it computes the following results: accuracy, precision,
		 recall, AUC, confusion matrix, F1, MAE, classification report.

		parameters:
		featureset = a data set containing a list of features
		y = the values of the test data. A string of 0's and 1's indicating the category in which an instance belongs.
		"""

		model = LogisticRegression()
		param_grid = {'C': [1,10], 'solver' : ['liblinear', 'lbfgs', 'saga']}
		cv = StratifiedKFold(n_splits = 10, random_state = 0)
		model = GridSearchCV(LogisticRegression(random_state = 0, max_iter = 900), param_grid, iid = True, cv = cv, scoring = 'f1')
		model.fit(train_f, y1)
		pan = model.best_params_
		ss = cross_val_score(model, train_f, y1, cv = cv, scoring = 'f1_weighted')
		pp = model.predict(test_f)
		acc = accuracy_score(y2, pp)
		conf = confusion_matrix(y2, pp)
		prec = precision_score(y2, pp)
		rec = recall_score(y2, pp)
		roc = roc_auc_score(y2, pp)
		MAE = mean_absolute_error(y2, pp)
		F1 = f1_score(y2, pp, average = "weighted")
		classy = classification_report(y2, pp)
		return('LR results:', pan, classy, 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )

