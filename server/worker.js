// Cloudflare Worker adaptation of the FastAPI backend
import { Hono } from 'hono'
import { cors } from 'hono/cors'

const app = new Hono()

// CORS configuration
const allowedOrigins = ['https://codesaviour.pages.dev', 'http://localhost:8015', 'http://localhost:8012']
app.use('*', cors({
  origin: (origin) => {
    const isPreview = origin?.startsWith('https://') && origin?.endsWith('.codesaviour.pages.dev')
    const isLocal = origin?.startsWith('http://localhost') || origin?.startsWith('http://127.0.0.1')
    if (origin && (allowedOrigins.includes(origin) || isPreview || isLocal)) {
      return origin
    }
    return '*'
  },
  allowHeaders: ['Content-Type', 'Authorization'],
  allowMethods: ['GET', 'POST', 'OPTIONS'],
}))

// Root endpoint to avoid 404
app.get('/', (c) => {
  return c.json({
    ok: true,
    message: 'CodeSaviour API running',
    endpoints: ['/api/status', 'POST /api/fix', 'POST /api/analyze']
  })
})

// Status endpoint
app.get('/api/status', (c) => {
  return c.json({
    fireworks_key_present: !!c.env.FIREWORKS_API_KEY,
    base_url: c.env.FIREWORKS_BASE_URL || "https://api.fireworks.ai/inference/v1",
    model: c.env.FIREWORKS_MODEL || "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
    site_url: c.env.SITE_URL || "https://codesaviour.pages.dev",
    openrouter_key_present: !!c.env.OPENROUTER_API_KEY,
    openrouter_base_url: c.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
    openrouter_model: c.env.OPENROUTER_MODEL || "qwen/qwen-2.5-coder-32b-instruct:free"
  })
})

// Fix endpoint
app.post('/api/fix', async (c) => {
  try {
    const { language, code, context } = await c.req.json()
    const original = code || ''

    if (!c.env.FIREWORKS_API_KEY) {
      return c.json({
        fixed: original,
        error: {
          type: "missing_api_key",
          detail: "FIREWORKS_API_KEY is not set",
          hint: "Add FIREWORKS_API_KEY as a Cloudflare Worker secret"
        }
      }, 401)
    }

    const prompt = `Fix the following ${language} code. Return only the corrected code without explanations:

\`\`\`${language}
${original}
\`\`\`

${context ? `Context: ${context}` : ''}`

    const url = `${c.env.FIREWORKS_BASE_URL || 'https://api.fireworks.ai/inference/v1'}/chat/completions`
    const model = c.env.FIREWORKS_MODEL || 'accounts/fireworks/models/qwen2p5-coder-32b-instruct'
    console.log('Fix: calling Fireworks', { url, model, len: original.length })

    let response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${c.env.FIREWORKS_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: "You are CodeSaviour. Return only corrected code." },
          { role: "user", content: prompt }
        ],
        temperature: 0,
        max_tokens: 2048
      })
    })

    if (!response.ok) {
      const errText = await response.text().catch(() => '')
      const isModel404 = response.status === 404 && /Model not found/i.test(errText || '')
      if (isModel404) {
        const fallbackModel = 'accounts/fireworks/models/llama-v3p1-8b-instruct'
        console.warn('Fix: primary model unavailable, retrying with fallback', { fallbackModel })
        response = await fetch(url, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${c.env.FIREWORKS_API_KEY}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: fallbackModel,
            messages: [
              { role: "system", content: "You are CodeSaviour. Return only corrected code." },
              { role: "user", content: prompt }
            ],
            temperature: 0,
            max_tokens: 2048
          })
        })
      } else {
        console.error('Fix: Fireworks error', { status: response.status, body: errText?.slice(0, 500) })
        return c.json({
          fixed: original,
          error: {
            type: 'fireworks_http_error',
            status: response.status,
            detail: errText?.slice(0, 500) || ''
          }
        }, response.status)
      }
    }

    const data = await response.json()
    const fixed = data.choices?.[0]?.message?.content?.trim() || original

    return c.json({ fixed })
  } catch (error) {
    console.error('Fix: handler failed', error?.message || String(error))
    const original = (typeof code === 'string' ? code : '')
    return c.json({
      fixed: original,
      error: {
        type: "fix_failed",
        detail: error.message,
        hint: "Check API configuration and network connectivity"
      }
    }, 502)
  }
})

// Analyze endpoint
app.post('/api/analyze', async (c) => {
  try {
    const { language, code } = await c.req.json()
    
    // Fallback to Fireworks if OpenRouter key is missing
    if (!c.env.OPENROUTER_API_KEY) {
      if (!c.env.FIREWORKS_API_KEY) {
        return c.json({
          errors: 0,
          warnings: 0,
          error_items: [],
          warning_items: [],
          info: 'Analysis unavailable: no OpenRouter or Fireworks API key configured'
        })
      }

      const fwUrl = `${c.env.FIREWORKS_BASE_URL || 'https://api.fireworks.ai/inference/v1'}/chat/completions`
      const fwModel = c.env.FIREWORKS_MODEL || 'accounts/fireworks/models/qwen2p5-coder-32b-instruct'
      console.log('Analyze: using Fireworks fallback', { url: fwUrl, model: fwModel, len: (code || '').length })

      const fwPrompt = `Analyze the following ${language} code for errors and warnings.\nReturn ONLY a valid JSON object with this exact structure and keys:\n{"errors": number, "warnings": number, "error_items": [{"line": number, "message": "string"}], "warning_items": [{"line": number, "message": "string"}]}\nNo extra text, code fences, or explanations.\n\nCode:\n\`\`\`${language}\n${code || ''}\n\`\`\``

      let fwResp = await fetch(fwUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${c.env.FIREWORKS_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: fwModel,
          messages: [
            { role: 'system', content: 'You are CodeSaviour. Return ONLY JSON with the specified keys.' },
            { role: 'user', content: fwPrompt }
          ],
          temperature: 0,
          max_tokens: 1024
        })
      })

      if (!fwResp.ok) {
        const errText = await fwResp.text().catch(() => '')
        const isModel404 = fwResp.status === 404 && /Model not found/i.test(errText || '')
        if (isModel404) {
          const fallbackModel = 'accounts/fireworks/models/llama-v3p1-8b-instruct'
          console.warn('Analyze: primary model unavailable, retrying with fallback', { fallbackModel })
          fwResp = await fetch(fwUrl, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${c.env.FIREWORKS_API_KEY}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              model: fallbackModel,
              messages: [
                { role: 'system', content: 'You are CodeSaviour. Return ONLY JSON with the specified keys.' },
                { role: 'user', content: fwPrompt }
              ],
              temperature: 0,
              max_tokens: 1024
            })
          })
        } else {
          console.error('Analyze: Fireworks error', { status: fwResp.status, body: errText?.slice(0, 500) })
          return c.json({
            errors: 0,
            warnings: 0,
            error_items: [],
            warning_items: [],
            info: `Fireworks analysis error: ${fwResp.status}`
          }, fwResp.status)
        }
      }

      const fwData = await fwResp.json()
      const content = fwData.choices?.[0]?.message?.content?.trim() || ''

      if (content) {
        try {
          const parsed = JSON.parse(content)
          return c.json({
            errors: parsed.errors || 0,
            warnings: parsed.warnings || 0,
            error_items: Array.isArray(parsed.error_items) ? parsed.error_items : [],
            warning_items: Array.isArray(parsed.warning_items) ? parsed.warning_items : []
          })
        } catch (e) {
          console.error('Analyze: Fireworks returned non-JSON content', content.slice(0, 200))
          return c.json({
            errors: 0,
            warnings: 0,
            error_items: [],
            warning_items: [],
            info: 'Analysis response was not valid JSON'
          })
        }
      }

      return c.json({
        errors: 0,
        warnings: 0,
        error_items: [],
        warning_items: [],
        info: 'Analysis response was empty'
      })
    }

    const prompt = `Analyze the following ${language} code for errors and warnings. Return a JSON object with this exact structure:
{
  "errors": number,
  "warnings": number,
  "error_items": [{"line": number, "message": "string"}],
  "warning_items": [{"line": number, "message": "string"}]
}

Code to analyze:
\`\`\`${language}
${code}
\`\`\``

    const base = c.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1'
    const openrouterModel = c.env.OPENROUTER_MODEL || 'qwen/qwen-2.5-coder-32b-instruct:free'
    const response = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${c.env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: openrouterModel,
        messages: [
          { role: 'system', content: 'You are CodeSaviour. Return ONLY JSON with the specified keys.'},
          { role: "user", content: prompt }
        ],
        temperature: 0
      })
    })

    if (!response.ok) {
      const errText = await response.text().catch(() => '')
      throw new Error(`Analysis API request failed: ${response.status} ${errText?.slice(0,300)}`)
    }

    const data = await response.json()
    const content = data.choices?.[0]?.message?.content?.trim()
    
    if (content) {
      try {
        const parsed = JSON.parse(content)
        return c.json({
          errors: parsed.errors || 0,
          warnings: parsed.warnings || 0,
          error_items: parsed.error_items || [],
          warning_items: parsed.warning_items || []
        })
      } catch (e) {
        console.warn('Analyze: OpenRouter returned non-JSON content, falling back to empty report')
        return c.json({
          errors: 0,
          warnings: 0,
          error_items: [],
          warning_items: [],
          info: 'Analysis response was not valid JSON'
        })
      }
    }

    return c.json({
      errors: 0,
      warnings: 0,
      error_items: [],
      warning_items: []
    })
  } catch (error) {
    return c.json({
      errors: 0,
      warnings: 0,
      error_items: [],
      warning_items: [],
      error: {
        type: "analysis_failed",
        detail: error.message
      }
    })
  }
})

export default app