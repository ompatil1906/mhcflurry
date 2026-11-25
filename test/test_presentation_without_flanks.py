# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
from unittest.mock import patch

from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor
from mhcflurry.class1_processing_predictor import Class1ProcessingPredictor


class DummyModel:
    sequence_lengths = {"n_flank": 0, "c_flank": 0, "peptide": 9}


class DummyAffinity(Class1AffinityPredictor):
    def __init__(self):
        super().__init__(allele_to_sequence={"HLA-A*02:01": "dummyseq"})

    def predict(self, peptides, alleles, **kwargs):
        return numpy.ones(len(peptides), dtype=float)


def test_predict_sequences_without_flanks_only_does_not_crash():
    """Regression: predictors that only have a without-flanks processing model
    should not raise when use_flanks defaults to True.
    """
    dummy_proc = Class1ProcessingPredictor(models=[DummyModel()])
    dummy_aff = DummyAffinity()
    predictor = Class1PresentationPredictor(
        affinity_predictor=dummy_aff,
        processing_predictor_without_flanks=dummy_proc,
        processing_predictor_with_flanks=None,
    )

    # Mock the heavy predict path so we only exercise the flanks selection logic
    def fake_predict(self, peptides, alleles, n_flanks=None, c_flanks=None, **kwargs):
        import pandas
        return pandas.DataFrame({
            "peptide": peptides,
            "affinity": numpy.ones(len(peptides)),
            "best_allele": ["HLA-A*02:01"] * len(peptides),
            "sample_name": ["sample"] * len(peptides),
            "peptide_num": range(len(peptides)),
        })

    with patch.object(Class1PresentationPredictor, "predict", fake_predict):
        result = predictor.predict_sequences(
            sequences={"p1": "ABCDEFGHIJKLMN"},
            alleles=["HLA-A*02:01"],
            verbose=0,
        )
    assert len(result) > 0
    assert "peptide" in result.columns
    # Crucially, it did not raise on the flanks path
