description = [
    {
        "description": "Perform rigid image registration between two medical images using SimpleITK. "
        "Rigid registration handles translation and rotation only, preserving shape and size. "
        "Includes preprocessing, similarity metrics calculation, and visualization generation.",
        "name": "quick_rigid_registration",
        "optional_parameters": [
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the image to be registered (moving image)",
                "name": "moving_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results and outputs",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform affine image registration between two medical images using SimpleITK. "
        "Affine registration handles translation, rotation, scaling, and shearing. "
        "More flexible than rigid registration but still preserves parallel lines.",
        "name": "quick_affine_registration",
        "optional_parameters": [
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the image to be registered (moving image)",
                "name": "moving_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results and outputs",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform deformable (B-spline) image registration between two medical images using SimpleITK. "
        "Deformable registration allows for local non-linear transformations, handling complex deformations. "
        "Most flexible but computationally intensive registration method.",
        "name": "quick_deformable_registration",
        "optional_parameters": [
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
            {
                "default": 4,
                "description": "Number of B-spline control points per dimension for deformable registration",
                "name": "number_of_control_points",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Path to the image to be registered (moving image)",
                "name": "moving_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results and outputs",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
    {
        "description": "Perform batch registration of multiple images to a single reference image. "
        "Automatically processes all medical image files in a directory and registers them to the fixed reference. "
        "Supports rigid, affine, or deformable registration for all images.",
        "name": "batch_register_images",
        "optional_parameters": [
            {
                "default": "rigid",
                "description": "Type of registration to perform: 'rigid', 'affine', or 'deformable'",
                "name": "transform_type",
                "type": "str",
            },
            {
                "default": "mutual_information",
                "description": "Similarity metric for registration: 'mutual_information', 'mean_squares', 'correlation', or 'normalized_correlation'",
                "name": "metric",
                "type": "str",
            },
            {
                "default": "gradient_descent",
                "description": "Optimization method: 'gradient_descent', 'lbfgsb', 'powell', or 'amoeba'",
                "name": "optimizer",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to preprocess images (denoising and normalization)",
                "name": "preprocess",
                "type": "bool",
            },
            {
                "default": True,
                "description": "Whether to create visualization plots for each registration",
                "name": "create_visualizations",
                "type": "bool",
            },
            {
                "default": 0.01,
                "description": "Learning rate for gradient descent optimizer",
                "name": "learning_rate",
                "type": "float",
            },
            {
                "default": 100,
                "description": "Maximum number of optimization iterations",
                "name": "number_of_iterations",
                "type": "int",
            },
            {
                "default": 1e-6,
                "description": "Convergence tolerance for optimization",
                "name": "gradient_convergence_tolerance",
                "type": "float",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Path to the reference (fixed) image file",
                "name": "fixed_image_path",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path containing multiple images to register (supports .nii, .nii.gz, .nrrd, .mha, .mhd formats)",
                "name": "moving_images_dir",
                "type": "str",
            },
            {
                "default": None,
                "description": "Directory path to save registration results for all images",
                "name": "output_dir",
                "type": "str",
            },
        ],
    },
]