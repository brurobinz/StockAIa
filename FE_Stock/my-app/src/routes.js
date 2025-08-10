import HomePage from './pages/HomePage';
import SignUpPage from './pages/SignUpPage';

const routes = [
    { path: '/', element: <HomePage /> },
    { path: '/signup', element: <SignUpPage /> },
];

export default routes;