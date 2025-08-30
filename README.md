# 📊 Data Visualizer

A powerful and interactive web application for data visualization built with Streamlit. This tool allows users to upload and visualize CSV data through various chart types including line plots, bar charts, scatter plots, distribution plots, and count plots.

## 🌟 Features

- **Interactive Data Selection**: Choose from pre-loaded CSV datasets or upload your own
- **Multiple Chart Types**: 
  - Line Plot
  - Bar Chart
  - Scatter Plot
  - Distribution Plot
  - Count Plot
- **Dynamic Axis Selection**: Select any columns as X and Y axes
- **Real-time Visualization**: Generate plots instantly with customizable parameters
- **Responsive Design**: Clean, modern interface that works on all devices
- **Pre-loaded Datasets**: Includes popular datasets for immediate exploration

## 📁 Included Datasets

The application comes with several sample datasets for immediate exploration:
- `diabetes.csv` - Diabetes dataset for medical analysis
- `heart.csv` - Heart disease dataset
- `parkinsons.csv` - Parkinson's disease dataset
- `tips.csv` - Restaurant tips dataset
- `titanic.csv` - Titanic passenger survival dataset

## 🚀 Live Demo

The application is deployed and available for public use. You can access it directly through the deployed link.

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Data_Visualizer.git
   cd Data_Visualizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Visualize_Data.py
   ```

4. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

1. **Select a Dataset**: Choose from the dropdown menu of available CSV files
2. **Preview Data**: View the first few rows of your selected dataset
3. **Choose Visualization**:
   - Select X-axis column from the dropdown
   - Select Y-axis column from the dropdown
   - Choose your preferred chart type
4. **Generate Plot**: Click the "Generate Plot" button to create your visualization
5. **Analyze Results**: Interact with the generated plot and explore your data

## 🎨 Supported Chart Types

- **Line Plot**: Perfect for showing trends over time or continuous relationships
- **Bar Chart**: Ideal for comparing categories or discrete data
- **Scatter Plot**: Great for exploring correlations between two variables
- **Distribution Plot**: Shows the distribution of a single variable with histogram and density curve
- **Count Plot**: Displays the frequency of categorical variables

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas
- **Visualization**: Matplotlib, Seaborn
- **Language**: Python

## 📦 Dependencies

- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `matplotlib` - Basic plotting library
- `seaborn` - Statistical data visualization

## 🚀 Deployment

This application is deployed using Streamlit Cloud. To deploy your own version:

1. **Push to GitHub**: Ensure your code is in a public GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Select your repository
   - Set the main file path to `Visualize_Data.py`
   - Click "Deploy"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

---

**Happy Data Visualization! 📊✨**