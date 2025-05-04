# VAE-image-reconstruction
VAE image reconstruction project on Fashion-MNIST
Overview
This project implements a Variational Autoencoder (VAE) to perform image reconstruction on the Fashion-MNIST dataset. The VAE is trained to encode images into a latent space and reconstruct them, providing insights into the model's ability to capture and regenerate visual features of different clothing categories. The project includes several visualizations to analyze the VAE's performance, such as reconstruction comparisons, latent space distributions, and class-wise reconstruction errors.
The Fashion-MNIST dataset consists of 60,000 training and 10,000 test grayscale images (28x28 pixels) across 10 clothing categories:

T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot

Project Objectives

Train a VAE: Learn latent representations of Fashion-MNIST images and reconstruct them.
Evaluate Performance: Compute Mean Squared Error (MSE) between original and reconstructed images, both overall and per class.
Visualize Results:
Side-by-side comparison of original vs. reconstructed images.
Latent space visualization to see how classes are distributed.
Class-wise MSE bar plot to compare reconstruction performance across categories.


Provide Insights: Interpret the results to understand which classes are reconstructed well or poorly.

Project Structure

vae_fashion_mnist.py: Main script containing the VAE model, training, evaluation, and visualization code.
requirements.txt: List of Python dependencies required to run the project.
data/: Directory where Fashion-MNIST data is automatically downloaded.
Output Files (generated after running the script):
reconstructions_comparison.png: Side-by-side comparison of original and reconstructed images.
latent_space.png: Scatter plot of the VAE's latent space, colored by class.
mse_bar_plot.png: Bar plot showing class-wise average MSE.
class_mse_summary.csv: CSV file with class-wise MSE values.



Outputs and Interpretation
1. Reconstruction Comparison (reconstructions_comparison.png)

Format: A 50x2 grid (50 rows, 2 columns).
Rows: Each row corresponds to one image sample (50 total: 5 samples per class, 10 classes).
Columns:
Column 1: Original image.
Column 2: Reconstructed image.


Labels:
Class name displayed above each row, colored by class.
"Original" and "Reconstructed" titles above the first row.


Styling: Images are grayscale with class-colored borders.


Interpretation:
Compare the original and reconstructed images to assess the VAE's reconstruction quality.
Clear reconstructions indicate the VAE has learned meaningful features; blurry or incorrect reconstructions suggest areas for improvement.



2. Latent Space Visualization (latent_space.png)

Format: A 2D scatter plot of latent variables (reduced via PCA if needed).
Points are colored by class, with a legend mapping colors to class names.


Interpretation:
Well-separated clusters indicate the VAE has learned distinct representations for each class.
Overlapping clusters suggest the VAE struggles to differentiate certain classes in the latent space.



3. Class-wise MSE Bar Plot (mse_bar_plot.png)

Format: A bar plot showing average MSE for each class, sorted from lowest to highest.
Bars are colored uniformly (skyblue) with value labels on top (e.g., 0.00012).
A horizontal grid aids comparison.
Y-axis ranges from 0 to 0.02 for context.
Title includes a note: "Lower MSE = Better Reconstruction".


Interpretation:
Lower MSE indicates better reconstruction performance for that class (e.g., Trouser might have the lowest MSE).
Higher MSE suggests poorer reconstruction (e.g., Bag might have the highest MSE), possibly due to complex patterns or variability.
Use the value labels to see exact MSE values without estimating from the y-axis.



4. Class-wise MSE Summary (class_mse_summary.csv)

Format: A CSV file with two columns: Class and Average MSE.
Interpretation:
Provides numerical data for the MSE bar plot.
Useful for detailed analysis or further processing.



5. Terminal Output

Prints the class-wise MSE table (sorted by MSE).
Includes an interpretation section explaining why some classes might be reconstructed better than others (e.g., simpler shapes like Trousers vs. complex patterns like Shirts).

Setup and Running Instructions
Prerequisites

GitHub Codespace or a local Python environment.
Python 3.12 or compatible version.
Git for version control.

Steps

Clone the Repository:

In your GitHub Codespace or local terminal:git clone <repository-url>
cd VAE-image-reconstruction




Install Dependencies:

Ensure requirements.txt contains:torch==2.4.1
torchvision==0.19.1
numpy==1.26.4
matplotlib==3.7.1
pandas==2.0.3
setuptools==70.0.0
scikit-learn==1.5.1


Install the dependencies:pip install -r requirements.txt




Run the Script:

Execute the main script to train the VAE and generate outputs:python vae_fashion_mnist.py


This will:
Train the VAE for 10 epochs (progress printed to terminal).
Download Fashion-MNIST data to the data/ directory.
Generate the visualizations and CSV file listed above.
Print class-wise MSE and interpretation to the terminal.




View Outputs:

Open the generated PNG files (reconstructions_comparison.png, latent_space.png, mse_bar_plot.png) to view visualizations.
Check class_mse_summary.csv for numerical results.
Review terminal output for additional insights.


Commit Changes (Optional):

If using GitHub Codespace, commit the generated files:git add *.png *.csv
git commit -m "Added VAE reconstruction outputs"
git push origin main





Customization and Troubleshooting
Customization

Training Epochs:
Increase the number of training epochs for better reconstruction quality (e.g., edit train_vae(epochs=20) in vae_fashion_mnist.py).
Note: More epochs increase training time.


Reconstruction Comparison:
The reconstructions_comparison.png has 50 rows (5 samples per class). To reduce the number of rows, modify num_samples in visualize_reconstructions (e.g., set to 1 for 10 rows).
Adjust figsize in visualize_reconstructions for a different aspect ratio (e.g., figsize=(8, 1.5 * total_images)).


MSE Bar Plot:
Change the bar color by editing color='skyblue' in visualize_mse_bar (e.g., color='lightgreen').
Adjust the y-axis range with plt.ylim(0, 0.03) if needed.



Troubleshooting

Module Not Found Error:
Ensure all dependencies are installed (pip install -r requirements.txt).
Verify Python version compatibility (3.12 recommended).


Poor Reconstruction Quality:
Increase training epochs (e.g., train_vae(epochs=20)).
Check the latent space visualization (latent_space.png) for class separation; poor separation may indicate the VAE needs more training or a different architecture.


Visualization Issues:
If reconstructions_comparison.png is too tall to view comfortably, reduce num_samples as mentioned above.
Increase dpi (e.g., dpi=600) in visualization functions for higher resolution, though this increases file size.



Dependencies

torch==2.4.1: For building and training the VAE.
torchvision==0.19.1: For loading the Fashion-MNIST dataset.
numpy==1.26.4: For numerical operations.
matplotlib==3.7.1: For generating visualizations.
pandas==2.0.3: For handling class-wise MSE data.
setuptools==70.0.0: Ensures build compatibility.
scikit-learn==1.5.1: For PCA in latent space visualization.

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as needed.
Contact
For questions or contributions, please open an issue on the GitHub repository or contact the project maintainer.
