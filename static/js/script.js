// AI MediDetect - Custom JavaScript

document.addEventListener('DOMContentLoaded', function() {

    // Form validation for prediction form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            const checkedSymptoms = document.querySelectorAll('input[name="symptoms"]:checked');
            if (checkedSymptoms.length === 0) {
                e.preventDefault();
                alert('Please select at least one symptom to proceed.');
                return false;
            }
        });
    }

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Console welcome message
    console.log('%c AI MediDetect ', 'background: #0d6efd; color: white; font-size: 20px; padding: 10px;');
    console.log('Welcome to AI MediDetect - AI-Based Disease Prediction System');
    console.log('Educational Project for MCA College Submission');
});
