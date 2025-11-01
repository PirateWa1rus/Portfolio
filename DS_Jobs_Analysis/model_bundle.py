## model_bundle.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class QuantilePredictor:
    preprocessor: Any            # fitted sklearn transformer / pipeline
    feature_names: Sequence[str]
    model_q10: Any               # fitted LGBMRegressor for alpha=0.1
    model_q50: Any               # fitted LGBMRegressor for alpha=0.5
    model_q90: Any               # fitted LGBMRegressor for alpha=0.9
    trained_on_log: bool = True  # True if you trained on log(y)
    metadata: dict = None

    def predict_quantiles(self, X_raw):
        # X_raw: pandas DataFrame with original columns (untransformed)
        X_trans = self.preprocessor.transform(X_raw)
        # ensure column ordering if needed:
        # X_trans_df = pd.DataFrame(X_trans, columns=self.feature_names)
        q10_log = self.model_q10.predict(X_trans)
        q50_log = self.model_q50.predict(X_trans)
        q90_log = self.model_q90.predict(X_trans)
        if self.trained_on_log:
            q10 = np.expm1(q10_log)
            q50 = np.expm1(q50_log)
            q90 = np.expm1(q90_log)
        else:
            q10, q50, q90 = q10_log, q50_log, q90_log
        return pd.DataFrame({"q10": q10, "q50": q50, "q90": q90})
