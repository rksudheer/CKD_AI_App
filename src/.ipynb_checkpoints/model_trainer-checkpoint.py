{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ab2ca346-0b8b-4f93-8825-d057ebcc95a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# CKD_AI_App/src/model_trainer.py\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import pandas as pd\n",
    "\n",
    "def train_logistic_regression(X_train, y_train):\n",
    "    model = LogisticRegression(max_iter=1000)\n",
    "    model.fit(X_train, y_train)\n",
    "    return model\n",
    "\n",
    "def evaluate_model(model, X_test, y_test):\n",
    "    predictions = model.predict(X_test)\n",
    "    report = classification_report(y_test, predictions, output_dict=True)\n",
    "    cm = confusion_matrix(y_test, predictions)\n",
    "\n",
    "    return {\n",
    "        \"classification_report\": pd.DataFrame(report).transpose(),\n",
    "        \"confusion_matrix\": cm\n",
    "    }"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
