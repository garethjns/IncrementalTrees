from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import classification_report


class Data:
    def _prep_data(self):
        x, y = make_blobs(n_samples=int(2e4),
                          random_state=0,
                          n_features=40,
                          centers=2,
                          cluster_std=60)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                                                test_size=0.25,
                                                                                random_state=123)

        return self

    def _mod_report(self, mod):

        report = classification_report(self.y_test, mod.predict(self.x_test))
        train_auc = roc_auc_score(self.y_train, mod.predict_proba(self.x_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, mod.predict_proba(self.x_test)[:, 1])

        return report, train_auc, test_auc
