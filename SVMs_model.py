	#SVMs
	def SVMs_model(train_f, test_f, y1, y2):
		""" This function creates the machine learning models for the Support Vector Machines method (a Bag of Words Principle).
		The function optimizes the model through Gridsearch. Then, it computes the following results: accuracy, precision,
		 recall, AUC, confusion matrix, F1, MAE, classification report.

		parameters:
		featureset = a data set containing a list of features
		y = the values of the test data. A string of 0's and 1's indicating the category to which an instance belongs.
		"""

		model = svm.SVC()
		param_grid = {'C':[1,10]}
		cv = StratifiedKFold(n_splits = 10, random_state = 0)
		model = GridSearchCV(svm.SVC(gamma = 'scale', kernel = 'linear'), param_grid, iid = True, cv = cv, refit = True, verbose = 2, scoring = 'f1')
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
		return('SVMs results:', pan, classy, 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )
