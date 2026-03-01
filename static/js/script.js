document.addEventListener("DOMContentLoaded", function() {
    const fillBar = document.getElementById('fill-bar');
    
    if (fillBar) {
        // Get the percentage from the data attribute
        const winProbability = fillBar.getAttribute('data-width');
        
        // Trigger animation after a slight delay
        setTimeout(() => {
            fillBar.style.width = winProbability + "%";
        }, 500);
    }
});