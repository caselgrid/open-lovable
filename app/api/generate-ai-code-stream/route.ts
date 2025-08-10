import { NextRequest, NextResponse } from 'next/server';
import { createGroq } from '@ai-sdk/groq';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createCerebras } from '@ai-sdk/cerebras';
import { streamText } from 'ai';
import type { ConversationState } from '@/types/conversation';

declare global {
  var conversationState: ConversationState | null;
}

const groq = createGroq({
  apiKey: process.env.GROQ_API_KEY,
});

const anthropic = createAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com/v1',
});

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const cerebras = createCerebras({
  apiKey: process.env.CEREBRAS_API_KEY || 'csk-thnmdvteer582myth45rvr4h4jnmx29tr6m8rcfj6e88hwkj',
});

export async function POST(request: NextRequest) {
  try {
    const { 
      prompt, 
      model = 'cerebras/qwen-3-235b-a22b-instruct-2507',
      systemPrompt,
      fileContents,
      isEdit = false,
      temperature = 0.7
    } = await request.json();
    
    if (!prompt) {
      return NextResponse.json({
        error: 'prompt is required'
      }, { status: 400 });
    }
    
    console.log('[generate-ai-code-stream] Request received');
    console.log('[generate-ai-code-stream] Model:', model);
    console.log('[generate-ai-code-stream] Is edit:', isEdit);
    console.log('[generate-ai-code-stream] Temperature:', temperature);
    
    // Select the appropriate AI model based on the request
    let aiModel;
    if (model.startsWith('anthropic/')) {
      aiModel = anthropic(model.replace('anthropic/', ''));
    } else if (model.startsWith('openai/')) {
      if (model.includes('gpt-oss')) {
        aiModel = groq(model);
      } else {
        aiModel = openai(model.replace('openai/', ''));
      }
    } else if (model.startsWith('cerebras/')) {
      aiModel = cerebras(model.replace('cerebras/', ''));
    } else {
      // Default to cerebras if model format is unclear
      aiModel = cerebras(model);
    }
    
    console.log('[generate-ai-code-stream] Using AI model:', model);
    
    // Build the system prompt
    let fullSystemPrompt = `You are an expert React developer. Create modern, responsive React applications using Vite, Tailwind CSS, and best practices.

CRITICAL RULES:
1. You MUST specify packages using <package> tags BEFORE using them in your code
2. For example: <package>three</package> or <package>@heroicons/react</package>
3. ALWAYS return complete files - never truncate with "..." or skip lines
4. Use Tailwind CSS for all styling (no separate CSS files unless specifically requested)
5. Create clean, modern, responsive designs
6. Use proper React patterns and hooks

Package Installation:
- Use <package>package-name</package> tags for individual packages
- Or use <packages>package1, package2, package3</packages> for multiple packages
- The system will automatically install these packages before creating files

File Structure:
- Always create files with proper paths (e.g., src/components/Header.jsx)
- Include all necessary imports
- Export components properly

${systemPrompt || ''}`;

    if (fileContents) {
      fullSystemPrompt += `\n\nCurrent project files:\n${fileContents}`;
    }
    
    // Track packages detected in real-time
    const packagesToInstall: string[] = [];
    let tagBuffer = '';
    
    // Create the stream
    const result = await streamText({
      model: aiModel,
      system: fullSystemPrompt,
      prompt,
      temperature,
      maxTokens: 8000,
    });
    
    // Create a response stream
    const encoder = new TextEncoder();
    const stream = new TransformStream();
    const writer = stream.writable.getWriter();
    
    // Function to send progress updates
    const sendProgress = async (data: any) => {
      const message = `data: ${JSON.stringify(data)}\n\n`;
      await writer.write(encoder.encode(message));
    };
    
    // Process the stream
    (async () => {
      try {
        let fullResponse = '';
        
        for await (const chunk of result.textStream) {
          const text = chunk;
          fullResponse += text;
          
          // Send the text chunk
          await sendProgress({
            type: 'text',
            content: text
          });
          
          // Buffer for tag detection across chunks
          const searchText = tagBuffer + text;
          
          // Detect individual package tags
          const packageRegex = /<package>([^<]+)<\/package>/g;
          let packageMatch;
          
          while ((packageMatch = packageRegex.exec(searchText)) !== null) {
            const packageName = packageMatch[1].trim();
            if (packageName && !packagesToInstall.includes(packageName)) {
              packagesToInstall.push(packageName);
              await sendProgress({ 
                type: 'package', 
                name: packageName,
                message: `📦 Package detected: ${packageName}`
              });
            }
          }
          
          // Detect packages block
          const packagesBlockRegex = /<packages>([\s\S]*?)<\/packages>/g;
          let packagesMatch;
          
          while ((packagesMatch = packagesBlockRegex.exec(searchText)) !== null) {
            const packagesContent = packagesMatch[1].trim();
            const packagesList = packagesContent.split(/[\n,]+/)
              .map(pkg => pkg.trim())
              .filter(pkg => pkg.length > 0);
            
            for (const pkg of packagesList) {
              if (!packagesToInstall.includes(pkg)) {
                packagesToInstall.push(pkg);
                await sendProgress({ 
                  type: 'package', 
                  name: pkg,
                  message: `📦 Package detected: ${pkg}`
                });
              }
            }
          }
          
          // Update buffer for next iteration
          tagBuffer = searchText.slice(-100); // Keep last 100 chars for tag detection
        }
        
        // Send completion
        await sendProgress({
          type: 'complete',
          response: fullResponse,
          packages: packagesToInstall,
          message: `Generated ${fullResponse.length} characters with ${packagesToInstall.length} packages`
        });
        
        // Update conversation state if available
        if (global.conversationState) {
          global.conversationState.context.messages.push({
            id: `msg-${Date.now()}`,
            role: 'user',
            content: prompt,
            timestamp: Date.now(),
            metadata: {
              editType: isEdit ? 'edit' : 'generate',
              addedPackages: packagesToInstall
            }
          });
          
          global.conversationState.context.messages.push({
            id: `msg-${Date.now() + 1}`,
            role: 'assistant',
            content: fullResponse,
            timestamp: Date.now(),
            metadata: {
              addedPackages: packagesToInstall
            }
          });
          
          global.conversationState.lastUpdated = Date.now();
        }
        
      } catch (error) {
        await sendProgress({
          type: 'error',
          error: (error as Error).message
        });
      } finally {
        await writer.close();
      }
    })();
    
    // Return the stream
    return new Response(stream.readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
    
  } catch (error) {
    console.error('[generate-ai-code-stream] Error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to generate code' },
      { status: 500 }
    );
  }
}