"""Just example code as explanation. Usable for testing."""
from src.data import HCPDataContainer, ISMRMDataContainer
from src.data.postprocessing import res100, spherical_harmonics
from src.dataset import StreamlineDataset, ConcatenatedDataset
from src.tracker import CSDTracker, ISMRMReferenceStreamlinesTracker
from src.dataset.processing import RegressionProcessing, ClassificationProcessing
def main():
    """Main method"""
    hcp_data = HCPDataContainer(100307)
    hcp_sl = CSDTracker(hcp_data, random_seeds=True, seeds_count=10000)
    ismrm_data = ISMRMDataContainer()
    ismrm_sl = ISMRMReferenceStreamlinesTracker(ismrm_data, streamline_count=10000)
    print("Loaded DataContainers")
    hcp_sl.track()
    ismrm_sl.track()
    print("Tracked Streamlines")
    ismrm_data = ismrm_data.normalize().crop()
    hcp_data = hcp_data.normalize().crop()
    print("Normalized and cropped data")
    processing = RegressionProcessing(postprocessing=res100())
    csd = StreamlineDataset(hcp_sl, hcp_data, processing)
    ismrm_set = StreamlineDataset(ismrm_sl, ismrm_data, processing)

    dataset = ConcatenatedDataset([csd, ismrm_set])
    print("Initialised Regression ")
    processing = ClassificationProcessing(postprocessing=spherical_harmonics())
    csd_classification = StreamlineDataset(hcp_sl, hcp_data, processing)
    ismrm_classification = StreamlineDataset(ismrm_sl, ismrm_data, processing)
    dataset_classification = ConcatenatedDataset([csd_classification, ismrm_classification])
    print("Initialised Classification Datasets")
if __name__ == "__main__":
    main()
