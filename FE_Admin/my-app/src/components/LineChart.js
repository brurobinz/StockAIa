import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import '../components/LineChart.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const LineChart = () => {
  const data = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
    datasets: [
      {
        label: 'Dataset 1',
        data: [30, 20, 50, 40, 60, 70, 80],
        borderColor: 'yellow',
        fill: false,
      },
      {
        label: 'Dataset 2',
        data: [20, 30, 40, 50, 20, 40, 60],
        borderColor: 'black',
        fill: false,
      },
      // Add more datasets as needed
    ],
  };

  return (
    <div style={{ width: '800px', height: '400px', margin: '0 auto' }}>
      <Line data={data} options={{ maintainAspectRatio: false }} />
    </div>
  );
};

export default LineChart;
