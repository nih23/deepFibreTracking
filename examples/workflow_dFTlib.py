"""Just example code as explanation. Usable for testing."""
 
from dfibert.data import DataPreprocessor
from dfibert.data.postprocessing import Resample100, SphericalHarmonics
from dfibert.dataset import StreamlineDataset, ConcatenatedDataset
from dfibert.tracker import get_csd_streamlines, save_streamlines, load_streamlines, get_dti_streamlines, filtered_streamlines_by_length
from dfibert.dataset.processing import RegressionProcessing, ClassificationProcessing


def main():
    """Main method"""
    hcp_data = DataPreprocessor().get_hcp("data/HCP/100307")
    ismrm_data = DataPreprocessor().get_ismrm("path/to/ismrm")
    print("Loaded DataContainers")
    hcp_sl = get_dti_streamlines(hcp_data, random_seeds=True, seeds_count=10000)

    save_streamlines(hcp_sl, "sls3.vtk")
    save_streamlines(filtered_streamlines_by_length(hcp_sl), "sls4.vtk")
    ismrm_sl = load_streamlines("path/to/ismrm/ground_truth")
    print("Tracked Streamlines")
    preprocessor = DataPreprocessor().normalize().fa_estimate().crop()

    ismrm_data = preprocessor.preprocess(ismrm_data)
    hcp_data = preprocessor.preprocess(hcp_data)

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
