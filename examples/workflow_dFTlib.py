"""Just example code as explanation. Usable for testing."""
from dfibert.data import DataPreprocessor
from dfibert.data.postprocessing import Resample100, SphericalHarmonics
from dfibert.dataset import StreamlineDataset, ConcatenatedDataset
from dfibert.tracker import CSDTracker, ISMRMReferenceStreamlinesTracker
from dfibert.dataset.processing import RegressionProcessing, ClassificationProcessing


def main():
    """Main method"""
    preprocessor = DataPreprocessor().normalise().fa_estimate().crop()
    hcp_data = preprocessor.get_hcp("path/to/hcp/")
    hcp_sl = CSDTracker(hcp_data, random_seeds=True, seeds_count=10000)
    ismrm_data = preprocessor.get_ismrm("path/to/ismrm")
    ismrm_sl = ISMRMReferenceStreamlinesTracker(ismrm_data, streamline_count=10000)
    print("Loaded DataContainers")
    hcp_sl.track()
    ismrm_sl.track()
    print("Tracked Streamlines")
    ismrm_data = ismrm_data.normalize().crop()
    hcp_data = hcp_data.normalize().crop()
    print("Normalized and cropped data")
    processing = RegressionProcessing(postprocessing=Resample100())
    csd = StreamlineDataset(hcp_sl, hcp_data, processing)
    ismrm_set = StreamlineDataset(ismrm_sl, ismrm_data, processing)

    dataset = ConcatenatedDataset([csd, ismrm_set])
    print("Initialised Regression ")
    processing = ClassificationProcessing(postprocessing=SphericalHarmonics())
    csd_classification = StreamlineDataset(hcp_sl, hcp_data, processing)
    ismrm_classification = StreamlineDataset(ismrm_sl, ismrm_data, processing)
    dataset_classification = ConcatenatedDataset([csd_classification, ismrm_classification])
    print("Initialised Classification Datasets")


if __name__ == "__main__":
    main()
