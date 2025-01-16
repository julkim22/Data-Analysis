# Student Progress Tracker

A project for tracking and analyzing student progress based on their `total_words` and `unique_words` metrics. This system helps educators and parents monitor and improve student performance by providing meaningful insights into their vocabulary usage.

---

## Features

- **Real-time Progress Tracking**: Monitors student activity and tracks `total_words` and `unique_words` metrics.
- **Customizable Notifications**: Sends alerts based on recent performance metrics to ensure timely interventions.
- **Detailed Insights**: Provides maximum, average, and recent statistics for more actionable data.
- **User-friendly Interface**: Simplifies data visualization and accessibility for educators and parents.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/julkim22
   ```
2. Navigate to the project directory:
   ```bash
   cd student-progress-tracker
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the System

1. Launch the application:
   ```bash
   python main.py
   ```
2. Access the dashboard to view student progress and receive notifications.

### Configuring Alerts

- Adjust thresholds for notifications in the configuration file:
  ```yaml
  notification_thresholds:
    total_words: 1000
    unique_words: 300
  ```
- Enable or disable specific features in the `settings.json` file.

---

## Project Structure

```
student-progress-tracker/
├── data/               # Sample datasets and logs
├── src/                # Source code for the application
│   ├── analysis.py     # Analysis and processing logic
│   ├── notifications.py # Notification system
│   ├── visualization.py # Visualization tools
├── notebooks/          # Jupyter Notebooks for development
├── tests/              # Unit and integration tests
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## Key Metrics

- **Total Words**: The cumulative number of words used by the student.
- **Unique Words**: The distinct number of words used, showcasing vocabulary diversity.
- **Recent Trends**: Analyzes data from the latest sessions for actionable insights.
- **Max Performance**: Highlights peak achievements in total and unique words.

---

## Example Output

```plaintext
Student: Jane Doe
- Total Words: 1500
- Unique Words: 450
- Recent Average (7 lessons):
  - Total Words: 1300
  - Unique Words: 400
Notifications:
- Excellent progress in the last week! Keep it up.
- Suggest reviewing recent performance to maintain consistency.
```
---

## Contact

For questions or feedback, reach out to [kito59@gmail.com](mailto: kito59@gmail.com).
