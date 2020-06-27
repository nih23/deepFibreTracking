"""Just example code as explanation. Usable for testing."""
from src.data import HCPDataContainer, ISMRMDataContainer
from src.data.postprocessing import res100, spherical_harmonics
from src.dataset import StreamlineDataset, ConcatenatedDataset, StreamlineClassificationDataset
from src.tracker import CSDTracker, ISMRMReferenceStreamlinesTracker

def main():
    """Main method"""
    hcp_data = HCPDataContainer(100307)
    hcp_sl = CSDTracker(hcp_data, random_seeds=True, seeds_count=10000)
    ismrm_data = ISMRMDataContainer()
    ismrm_sl = ISMRMReferenceStreamlinesTracker(ismrm_data, streamline_count=10000)
    hcp_sl.track()
    ismrm_sl.track()
    ismrm_data = ismrm_data.crop().normalize()
    hcp_data = hcp_data.crop().normalize()
    csd = StreamlineDataset(hcp_sl, hcp_data, postprocessing=res100())
    ismrm_set = StreamlineDataset(ismrm_sl, ismrm_data, postprocessing=res100())
    dataset = ConcatenatedDataset([csd, ismrm_set])

    csd_classification = StreamlineClassificationDataset(hcp_sl, hcp_data,
                                                         postprocessing=spherical_harmonics())
    ismrm_classification = StreamlineClassificationDataset(ismrm_sl, ismrm_data,
                                                           postprocessing=spherical_harmonics())
    dataset_classification = ConcatenatedDataset([csd_classification, ismrm_classification])

if __name__ == "__main__":
    main()
