# CodeSaviour Cloudflare Deployment Guide

## Prerequisites

1. Install Cloudflare CLI (Wrangler):
   ```bash
   npm install -g wrangler
   ```

2. Login to Cloudflare:
   ```bash
   wrangler login
   ```

## Deployment Steps

### 1. Deploy the Backend API (Cloudflare Workers)

```bash
cd server
npm install
wrangler secret put FIREWORKS_API_KEY
wrangler secret put OPENROUTER_API_KEY
wrangler deploy
```

After deployment, note your Worker URL (e.g., `https://codesaviour-api.your-subdomain.workers.dev`)

### 2. Update Frontend Configuration

Edit `dashboard.js` and replace `your-subdomain` with your actual Worker subdomain:
```javascript
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
  ? 'http://127.0.0.1:8001' 
  : 'https://codesaviour-api.YOUR-ACTUAL-SUBDOMAIN.workers.dev';
```

### 3. Deploy the Frontend (Cloudflare Pages)

```bash
npm install
npm run build
wrangler pages deploy dist --project-name codesaviour
```

## Environment Variables

Set these secrets in your Cloudflare Worker:

- `FIREWORKS_API_KEY`: Your Fireworks AI API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key (optional, for enhanced analysis)

## Custom Domain (Optional)

1. In Cloudflare Dashboard, go to Workers & Pages
2. Select your deployed sites
3. Go to Custom domains and add your domain
4. Update the CORS origins in `worker.js` to include your custom domain

## Verification

After deployment:
1. Visit your Cloudflare Pages URL
2. Test the "Fix code" functionality
3. Test the "Run deep scan" functionality
4. Check the browser console for any errors

## Troubleshooting

- If you get CORS errors, ensure your Worker URL is correctly set in `dashboard.js`
- If API calls fail, check that your secrets are properly set in the Worker
- For custom domains, update the CORS configuration in `worker.js`