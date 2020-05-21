import warnings

import numpy as np
from sklearn.utils.multiclass import unique_labels


def _check_partial_fit_first_call(clf,
                                  classes=None):
    """
    Modified sklearn function. If classes are inconsistent on second call, warn and reuse previous
    on assumption first call specification was correct. Don't raise error.

    Private helper function for factorizing common classes param logic

    Estimators that implement the ``partial_fit`` API need to be provided with
    the list of possible classes at the first call to partial_fit.

    Modification:
    Subsequent calls to partial_fit do not check () that ``classes`` is still
    consistent with a previous value of ``clf.classes_`` when provided.

    This function returns True if it detects that this was the first call to
    ``partial_fit`` on ``clf``. In that case the ``classes_`` attribute is also
    set on ``clf``.
    """

    if getattr(clf, 'classes_', None) is None and classes is None:
        raise ValueError("classes must be passed on the first call "
                         "to partial_fit.")

    elif classes is not None:
        if getattr(clf, 'classes_', None) is not None:
            if not np.array_equal(clf.classes_, unique_labels(classes)):
                # Don't error here:
                # Instead, use the previous classes setting, which must be correct on first setting
                warnings.warn(f"Classes differ on this call, ignoring on the assumption first call was correct.")
                return False
                # raise ValueError(
                #     "`classes=%r` is not the same as on last call "
                #     "to partial_fit, was: %r" % (classes, clf.classes_))

        else:
            # This is the first call to partial_fit
            clf.classes_ = unique_labels(classes)
            return True

    # classes is None and clf.classes_ has already previously been set:
    # nothing to do
    return False
