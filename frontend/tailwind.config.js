/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        indigo: {
          night: '#12182B',
          deep: '#1B2340',
        },
        paper: {
          DEFAULT: '#EDEFF3',
          card: '#F7F8FA',
        },
        amber: {
          board: '#F2A93B',
          bright: '#FFC868',
        },
        stamp: {
          red: '#B23A2F',
        },
        teal: {
          promo: '#2F8F7B',
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['"IBM Plex Sans"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
