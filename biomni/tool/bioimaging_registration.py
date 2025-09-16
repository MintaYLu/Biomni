"""
Bioimaging Registration Tool using SimpleITK

This module provides comprehensive image registration capabilities for bioimaging data,
including rigid, affine, and deformable registration using SimpleITK.

Author: Biomni Team
"""

import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRegistrationTool:
    """
    A comprehensive image registration tool using SimpleITK.
    
    This class provides methods for various types of image registration including
    rigid, affine, and deformable registration with multiple similarity metrics
    and optimization strategies.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the ImageRegistrationTool.
    
    Args:
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
    
    def load_image(self, file_path: Union[str, Path]) -> sitk.Image:
        """
        Load an image using SimpleITK.
    
    Args:
            file_path: Path to the image file
        
    Returns:
            SimpleITK Image object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If the image cannot be loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            image = sitk.ReadImage(str(file_path))
            logger.info(f"Successfully loaded image: {file_path}")
            logger.info(f"Image size: {image.GetSize()}")
            logger.info(f"Image spacing: {image.GetSpacing()}")
            logger.info(f"Image origin: {image.GetOrigin()}")
            logger.info(f"Image direction: {image.GetDirection()}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            raise RuntimeError(f"Failed to load image: {e}")
    
    def save_image(self, image: sitk.Image, output_path: Union[str, Path]) -> Path:
        """
        Save a SimpleITK image.
    
    Args:
            image: SimpleITK Image object
            output_path: Path to save the image
        
    Returns:
            Path to saved file
            
        Raises:
            RuntimeError: If the image cannot be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            sitk.WriteImage(image, str(output_path))
            logger.info(f"Successfully saved image: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save image {output_path}: {e}")
            raise RuntimeError(f"Failed to save image: {e}")
    
    def preprocess_image(self, image: sitk.Image, 
                        normalize: bool = True,
                        denoise: bool = True,
                        resample_spacing: Optional[Tuple[float, ...]] = None) -> sitk.Image:
        """
        Preprocess an image for better registration.
    
    Args:
            image: Input image
            normalize: Whether to normalize intensity values
            denoise: Whether to apply denoising
            resample_spacing: Target spacing for resampling (optional)
        
    Returns:
            Preprocessed image
        """
        logger.info("Preprocessing image...")
        processed_image = sitk.Image(image)
        
        # Resample to target spacing if specified
        if resample_spacing is not None:
            logger.info(f"Resampling to spacing: {resample_spacing}")
            original_spacing = processed_image.GetSpacing()
            original_size = processed_image.GetSize()
            
            new_size = [int(original_size[i] * original_spacing[i] / resample_spacing[i]) 
                       for i in range(processed_image.GetDimension())]
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(resample_spacing)
            resampler.SetSize(new_size)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputOrigin(processed_image.GetOrigin())
            resampler.SetOutputDirection(processed_image.GetDirection())
            
            processed_image = resampler.Execute(processed_image)
        
        # Denoise if requested
        if denoise:
            logger.info("Applying denoising...")
            # Use Gaussian smoothing for denoising
            gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
            gaussian_filter.SetSigma(0.5)  # Adjust sigma as needed
            processed_image = gaussian_filter.Execute(processed_image)
        
        # Normalize if requested
        if normalize:
            logger.info("Normalizing intensity values...")
            # Use histogram equalization for normalization
            processed_image = sitk.AdaptiveHistogramEqualization(processed_image)
        
        logger.info("Image preprocessing completed")
        return processed_image
    
    def create_initial_transform(self, fixed_image: sitk.Image, 
                               moving_image: sitk.Image,
                               transform_type: str = 'rigid') -> sitk.Transform:
        """
        Create an initial transformation for registration.
    
    Args:
            fixed_image: Reference image
            moving_image: Image to be registered
            transform_type: Type of transformation ('rigid', 'affine', 'deformable')
        
    Returns:
            Initial transformation
        """
        if transform_type == 'rigid':
            # Create identity transform for rigid registration
            transform = sitk.Transform()
        elif transform_type == 'affine':
            # Create affine transform
            transform = sitk.AffineTransform(fixed_image.GetDimension())
        elif transform_type == 'deformable':
            # Create BSpline transform for deformable registration
            transform_domain_physical_dimensions = [
                fixed_image.GetSpacing()[i] * fixed_image.GetSize()[i] 
                for i in range(fixed_image.GetDimension())
            ]
            bspline_order = 3
            transform = sitk.BSplineTransformInitializer(
                fixed_image, 
                transform_domain_physical_dimensions, 
                bspline_order
            )
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        return transform
    
    def setup_registration_method(self, 
                                 metric: str = 'mutual_information',
                                 optimizer: str = 'gradient_descent',
                                 interpolator: str = 'linear',
                                 number_of_histogram_bins: int = 50,
                                 learning_rate: float = 1.0,
                                 number_of_iterations: int = 200,
                                 relaxation_factor: float = 0.5,
                                 gradient_convergence_tolerance: float = 1e-6,
                                 sampling_strategy: str = 'random',
                                 sampling_percentage: float = 0.1) -> sitk.ImageRegistrationMethod:
        """
        Setup the registration method with specified parameters.
    
    Args:
            metric: Similarity metric ('mutual_information', 'mean_squares', 'correlation')
            optimizer: Optimization method ('gradient_descent', 'lbfgsb', 'powell')
            interpolator: Interpolation method ('linear', 'nearest_neighbor', 'bspline')
            number_of_histogram_bins: Number of histogram bins for mutual information
            learning_rate: Learning rate for optimization
            number_of_iterations: Maximum number of iterations
            relaxation_factor: Relaxation factor for optimization
            gradient_convergence_tolerance: Tolerance for gradient descent
            sampling_strategy: Sampling strategy ('random', 'regular', 'none')
            sampling_percentage: Percentage of pixels to sample
        
    Returns:
            Configured registration method
        """
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set similarity metric
        if metric == 'mutual_information':
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_of_histogram_bins)
        elif metric == 'mean_squares':
            registration_method.SetMetricAsMeanSquares()
        elif metric == 'correlation':
            registration_method.SetMetricAsCorrelation()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Set interpolator
        if interpolator == 'linear':
            registration_method.SetInterpolator(sitk.sitkLinear)
        elif interpolator == 'nearest_neighbor':
            registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
        elif interpolator == 'bspline':
            registration_method.SetInterpolator(sitk.sitkBSpline)
        else:
            raise ValueError(f"Unknown interpolator: {interpolator}")
        
        # Set optimizer
        if optimizer == 'gradient_descent':
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=learning_rate,
                numberOfIterations=number_of_iterations,
                convergenceMinimumValue=gradient_convergence_tolerance
            )
        elif optimizer == 'lbfgsb':
            registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=gradient_convergence_tolerance,
                numberOfIterations=number_of_iterations
            )
        elif optimizer == 'powell':
            registration_method.SetOptimizerAsPowell(
                numberOfIterations=number_of_iterations,
                maximumStepLength=learning_rate,
                minimumStepLength=gradient_convergence_tolerance
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Set optimizer scaling
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Set sampling
        if sampling_strategy == 'random':
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        elif sampling_strategy == 'regular':
            registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
        elif sampling_strategy == 'none':
            registration_method.SetMetricSamplingStrategy(registration_method.NONE)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        registration_method.SetMetricSamplingPercentage(sampling_percentage)
        
        return registration_method
    
    def register_images(self, 
                       fixed_image: sitk.Image, 
                       moving_image: sitk.Image,
                       transform_type: str = 'rigid',
                       metric: str = 'mutual_information',
                       optimizer: str = 'gradient_descent',
                       preprocess: bool = True,
                       **kwargs) -> Tuple[sitk.Transform, float, List[float]]:
        """
        Register two images using SimpleITK.
    
    Args:
            fixed_image: Reference (fixed) image
        moving_image: Image to be registered
            transform_type: Type of registration ('rigid', 'affine', 'deformable')
            metric: Similarity metric
            optimizer: Optimization method
            preprocess: Whether to preprocess images
            **kwargs: Additional parameters for registration setup
        
    Returns:
            Tuple of (final_transform, final_metric_value, metric_values_history)
        """
        logger.info(f"Starting {transform_type} registration...")
        logger.info(f"Fixed image size: {fixed_image.GetSize()}")
        logger.info(f"Moving image size: {moving_image.GetSize()}")
        
        # Preprocess images if requested
        if preprocess:
            fixed_image = self.preprocess_image(fixed_image)
            moving_image = self.preprocess_image(moving_image)
        
        # Create initial transform
        initial_transform = self.create_initial_transform(fixed_image, moving_image, transform_type)
        
        # Setup registration method
        registration_method = self.setup_registration_method(
            metric=metric,
            optimizer=optimizer,
            **kwargs
        )
        
        # Set initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # Add observer for monitoring progress
        metric_values = []
        def command_iteration():
            metric_values.append(registration_method.GetMetricValue())
            if self.verbose:
                logger.info(f"Iteration {len(metric_values)}: {registration_method.GetMetricValue():.4f}")
        
        registration_method.AddCommand(sitk.sitkIterationEvent, command_iteration)
        
        # Execute registration
        try:
            logger.info("Executing registration...")
            final_transform = registration_method.Execute(fixed_image, moving_image)
            final_metric_value = registration_method.GetMetricValue()
            
            logger.info(f"Registration completed successfully!")
            logger.info(f"Final metric value: {final_metric_value:.4f}")
            logger.info(f"Number of iterations: {len(metric_values)}")
            
            return final_transform, final_metric_value, metric_values
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            raise RuntimeError(f"Registration failed: {e}")
    
    def apply_transform(self, 
                       moving_image: sitk.Image, 
                       transform: sitk.Transform,
                       reference_image: Optional[sitk.Image] = None,
                       interpolator: str = 'linear') -> sitk.Image:
        """
        Apply a transformation to an image.
    
    Args:
            moving_image: Image to transform
            transform: Transformation to apply
            reference_image: Reference image for output space (optional)
            interpolator: Interpolation method
        
    Returns:
            Transformed image
        """
        logger.info("Applying transformation...")
        
        # Create resampler
        resampler = sitk.ResampleImageFilter()
        
        # Set reference image
        if reference_image is not None:
            resampler.SetReferenceImage(reference_image)
        else:
            resampler.SetReferenceImage(moving_image)
        
        # Set interpolator
        if interpolator == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interpolator == 'nearest_neighbor':
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif interpolator == 'bspline':
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            raise ValueError(f"Unknown interpolator: {interpolator}")
        
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
            
            # Apply transformation
        try:
            transformed_image = resampler.Execute(moving_image)
            logger.info("Transformation applied successfully")
            return transformed_image
        except Exception as e:
            logger.error(f"Failed to apply transformation: {e}")
            raise RuntimeError(f"Failed to apply transformation: {e}")
    
    def calculate_similarity_metrics(self, 
                                   image1: sitk.Image, 
                                   image2: sitk.Image) -> Dict[str, float]:
        """
        Calculate various similarity metrics between two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Dictionary of similarity metrics
        """
        logger.info("Calculating similarity metrics...")
        
        metrics = {}
        
        # Convert images to numpy arrays for calculation
        try:
            array1 = sitk.GetArrayFromImage(image1)
            array2 = sitk.GetArrayFromImage(image2)
            
            # Flatten arrays for correlation calculations
            flat1 = array1.flatten()
            flat2 = array2.flatten()
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(flat1) & np.isfinite(flat2)
            flat1 = flat1[valid_mask]
            flat2 = flat2[valid_mask]
            
            if len(flat1) == 0:
                logger.warning("No valid pixels for similarity calculation")
                return {'mutual_information': 0.0, 'mean_squares': 0.0, 
                       'correlation': 0.0, 'normalized_correlation': 0.0}
            
            # Mean Squared Error (lower is better, so we negate it)
            mse = np.mean((flat1 - flat2) ** 2)
            metrics['mean_squares'] = -mse  # Negate so higher is better
            
            # Pearson Correlation
            if np.std(flat1) > 0 and np.std(flat2) > 0:
                correlation = np.corrcoef(flat1, flat2)[0, 1]
                metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
            else:
                metrics['correlation'] = 0.0
            
            # Normalized Cross Correlation
            if np.std(flat1) > 0 and np.std(flat2) > 0:
                ncc = np.corrcoef(flat1, flat2)[0, 1]
                metrics['normalized_correlation'] = ncc if not np.isnan(ncc) else 0.0
            else:
                metrics['normalized_correlation'] = 0.0
            
            # Mutual Information (simplified calculation)
            try:
                # Use histogram-based mutual information calculation
                hist_2d, x_edges, y_edges = np.histogram2d(flat1, flat2, bins=50)
                hist_2d = hist_2d + 1e-10  # Add small value to avoid log(0)
                
                # Normalize to get probabilities
                pxy = hist_2d / np.sum(hist_2d)
                px = np.sum(pxy, axis=1)
                py = np.sum(pxy, axis=0)
                
                # Calculate mutual information
                mi = 0.0
                for i in range(len(px)):
                    for j in range(len(py)):
                        if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                            mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
                
                metrics['mutual_information'] = mi
            except Exception as e:
                logger.warning(f"Failed to calculate mutual information: {e}")
                metrics['mutual_information'] = 0.0
                
        except Exception as e:
            logger.warning(f"Failed to calculate similarity metrics: {e}")
            metrics = {'mutual_information': 0.0, 'mean_squares': 0.0, 
                      'correlation': 0.0, 'normalized_correlation': 0.0}
        
        logger.info("Similarity metrics calculated")
        return metrics
    
    def create_registration_visualization(self, 
                                        fixed_image: sitk.Image,
                                        moving_image: sitk.Image,
                                        registered_image: sitk.Image,
                                        output_dir: Union[str, Path],
                                        title_prefix: str = "Registration") -> List[Path]:
        """
        Create visualization of registration results.
    
    Args:
            fixed_image: Reference image
            moving_image: Original moving image
            registered_image: Registered image
        output_dir: Directory to save visualizations
            title_prefix: Prefix for plot titles
        
    Returns:
        List of saved visualization file paths
    """
        logger.info("Creating registration visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        try:
            # Convert SimpleITK images to numpy arrays for visualization
            fixed_array = sitk.GetArrayFromImage(fixed_image)
            moving_array = sitk.GetArrayFromImage(moving_image)
            registered_array = sitk.GetArrayFromImage(registered_image)
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{title_prefix} Results Comparison', fontsize=16)
            
            # Get middle slices for visualization
            mid_slice = fixed_array.shape[0] // 2
        
        # Fixed image
            axes[0, 0].imshow(fixed_array[mid_slice], cmap='gray')
            axes[0, 0].set_title('Fixed Image')
            axes[0, 0].axis('off')
        
        # Moving image
            axes[0, 1].imshow(moving_array[mid_slice], cmap='gray')
            axes[0, 1].set_title('Moving Image (Original)')
            axes[0, 1].axis('off')
        
        # Registered image
            axes[0, 2].imshow(registered_array[mid_slice], cmap='gray')
            axes[0, 2].set_title('Registered Image')
            axes[0, 2].axis('off')
            
            # Difference images
            diff_original = np.abs(fixed_array - moving_array)
            diff_registered = np.abs(fixed_array - registered_array)
            
            axes[1, 0].imshow(diff_original[mid_slice], cmap='hot')
            axes[1, 0].set_title('Difference (Fixed - Moving)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(diff_registered[mid_slice], cmap='hot')
            axes[1, 1].set_title('Difference (Fixed - Registered)')
            axes[1, 1].axis('off')
            
            # Overlay
            overlay = np.zeros((*fixed_array[mid_slice].shape, 3))
            overlay[:, :, 0] = fixed_array[mid_slice] / np.max(fixed_array)
            overlay[:, :, 1] = registered_array[mid_slice] / np.max(registered_array)
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Overlay (Red: Fixed, Green: Registered)')
            axes[1, 2].axis('off')
        
            # Save comparison plot
            comparison_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            saved_files.append(comparison_path)
            plt.close()
        
            # Create metric plots
            metrics = self.calculate_similarity_metrics(fixed_image, registered_image)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax.bar(metric_names, metric_values)
            ax.set_title(f'{title_prefix} Similarity Metrics')
            ax.set_ylabel('Metric Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            metrics_path = output_dir / f"{title_prefix.lower().replace(' ', '_')}_metrics.png"
            plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
            saved_files.append(metrics_path)
            plt.close()
        
            logger.info(f"Visualizations saved to: {output_dir}")
            for file_path in saved_files:
                logger.info(f"  - {file_path.name}")
        
            return saved_files
        
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
        return []

    def register_and_save(self, 
                         fixed_image_path: Union[str, Path],
                         moving_image_path: Union[str, Path],
                         output_dir: Union[str, Path],
                         transform_type: str = 'rigid',
                         metric: str = 'mutual_information',
                         optimizer: str = 'gradient_descent',
                         preprocess: bool = True,
                         create_visualizations: bool = True,
                         **kwargs) -> Dict[str, Any]:
        """
        Complete registration workflow: load, register, save, and visualize.
    
    Args:
            fixed_image_path: Path to reference image
            moving_image_path: Path to image to register
        output_dir: Directory to save results
            transform_type: Type of registration
            metric: Similarity metric
            optimizer: Optimization method
            preprocess: Whether to preprocess images
            create_visualizations: Whether to create visualizations
            **kwargs: Additional parameters for registration
        
    Returns:
            Dictionary with registration results
        """
        logger.info("Starting complete registration workflow...")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load images
            logger.info("Loading images...")
            fixed_image = self.load_image(fixed_image_path)
            moving_image = self.load_image(moving_image_path)
            
            # Register images
            logger.info("Registering images...")
            transform, metric_value, metric_history = self.register_images(
                fixed_image, moving_image, transform_type, metric, optimizer, preprocess, **kwargs
            )
            
            # Apply transformation
            logger.info("Applying transformation...")
            registered_image = self.apply_transform(moving_image, transform, fixed_image)
            
            # Save results
            logger.info("Saving results...")
            registered_path = output_dir / f"registered_{transform_type}.nii.gz"
            transform_path = output_dir / f"transform_{transform_type}.tfm"
            
            self.save_image(registered_image, registered_path)
            sitk.WriteTransform(transform, str(transform_path))
            
            # Create visualizations
            visualization_files = []
            if create_visualizations:
                logger.info("Creating visualizations...")
                visualization_files = self.create_registration_visualization(
                    fixed_image, moving_image, registered_image, output_dir, 
                    f"{transform_type.title()} Registration"
                )
            
            # Prepare results
            results = {
                'registered_image_path': str(registered_path),
                'transform_path': str(transform_path),
                'metric_value': metric_value,
                'metric_history': metric_history,
                'transform_type': transform_type,
                'metric': metric,
                'optimizer': optimizer,
                'visualization_files': [str(f) for f in visualization_files]
            }
            
            logger.info("Registration workflow completed successfully!")
            logger.info(f"Results saved to: {output_dir}")
    
            return results

        except Exception as e:
            logger.error(f"Registration workflow failed: {e}")
            raise RuntimeError(f"Registration workflow failed: {e}")


# Convenience functions for easy usage
def quick_rigid_registration(fixed_image_path: Union[str, Path],
                           moving_image_path: Union[str, Path],
                           output_dir: Union[str, Path],
                           **kwargs) -> Dict[str, Any]:
    """
    Quick rigid registration between two images.
    
    Args:
        fixed_image_path: Path to reference image
        moving_image_path: Path to image to register
        output_dir: Directory to save results
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with registration results
    """
    tool = ImageRegistrationTool()
    return tool.register_and_save(
        fixed_image_path, moving_image_path, output_dir,
        transform_type='rigid', **kwargs
    )


def quick_affine_registration(fixed_image_path: Union[str, Path],
                            moving_image_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            **kwargs) -> Dict[str, Any]:
    """
    Quick affine registration between two images.
    
    Args:
        fixed_image_path: Path to reference image
        moving_image_path: Path to image to register
        output_dir: Directory to save results
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with registration results
    """
    tool = ImageRegistrationTool()
    return tool.register_and_save(
        fixed_image_path, moving_image_path, output_dir,
        transform_type='affine', **kwargs
    )


def quick_deformable_registration(fixed_image_path: Union[str, Path],
                                moving_image_path: Union[str, Path],
                                output_dir: Union[str, Path],
                                **kwargs) -> Dict[str, Any]:
    """
    Quick deformable registration between two images.
    
    Args:
        fixed_image_path: Path to reference image
        moving_image_path: Path to image to register
        output_dir: Directory to save results
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with registration results
    """
    tool = ImageRegistrationTool()
    return tool.register_and_save(
        fixed_image_path, moving_image_path, output_dir,
        transform_type='deformable', **kwargs
    )


def batch_register_images(fixed_image_path: Union[str, Path],
                         moving_images_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         transform_type: str = 'rigid',
                         **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Batch register multiple images to a fixed reference.
    
    Args:
        fixed_image_path: Path to reference image
        moving_images_dir: Directory containing images to register
        output_dir: Directory to save results
        transform_type: Type of registration
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with results for each image
    """
    logger.info(f"Starting batch {transform_type} registration...")
    
    moving_images_dir = Path(moving_images_dir)
    output_dir = Path(output_dir)
    
    # Find all image files
    image_extensions = {'.nii', '.nii.gz', '.nrrd', '.mha', '.mhd'}
    moving_files = [f for f in moving_images_dir.iterdir() 
                   if f.suffix.lower() in image_extensions or 
                      (f.suffix == '.gz' and f.stem.endswith('.nii'))]
    
    if not moving_files:
        raise ValueError(f"No image files found in {moving_images_dir}")
    
    logger.info(f"Found {len(moving_files)} images to register")
    
    tool = ImageRegistrationTool()
    results = {}
    
    for i, moving_file in enumerate(moving_files):
        logger.info(f"Processing {i+1}/{len(moving_files)}: {moving_file.name}")
        
        try:
            individual_output_dir = output_dir / moving_file.stem
            result = tool.register_and_save(
                fixed_image_path, moving_file, individual_output_dir,
                transform_type=transform_type, **kwargs
            )
            results[moving_file.stem] = result
            logger.info(f"✅ Successfully registered {moving_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register {moving_file.name}: {e}")
            results[moving_file.stem] = {'error': str(e)}
    
    # Summary
    successful = sum(1 for r in results.values() if 'error' not in r)
    failed = len(results) - successful
    
    logger.info(f"Batch registration completed!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
        
    return results
        

if __name__ == "__main__":
    # Example usage
    print("Bioimaging Registration Tool")
    print("This module provides comprehensive image registration capabilities.")
    print("Use the ImageRegistrationTool class or convenience functions for registration.")