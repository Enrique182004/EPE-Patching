/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Ensuring the default sans font is used for Inter if desired
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        'primary-blue': '#1D4ED8',
        'secondary-gray': '#E5E7EB',
      }
    },
  },
  plugins: [],
}
