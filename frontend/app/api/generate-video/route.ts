// app/api/generate-video/route.ts
import { NextRequest, NextResponse } from 'next/server';

// Use a very long timeout - 10 minutes
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
const API_TIMEOUT = 10 * 60 * 1000; // 10 minutes in milliseconds

export async function POST(request: NextRequest) {
  try {
    // Extract the request body
    const body = await request.json();
    
    console.log(`Attempting to connect to backend at ${BACKEND_URL}/generate-content/`);
    
    // Set up AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);
    
    try {
      // Call the external API with the long timeout
      const response = await fetch(`${BACKEND_URL}/generate-content/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: body.prompt,
          genre: body.genre,
          iterations: body.iterations,
          backgroundType: body.backgroundType,
          musicType: body.musicType,
          voiceType: body.voiceType,
          subtitleColor: body.subtitleColor,
        }),
        signal: controller.signal,
      });
      
      // Clear the timeout
      clearTimeout(timeoutId);
      
      // Check if the request was successful
      if (!response.ok) {
        // Get the error details from the response if possible
        let errorDetail = 'Unknown error';
        try {
          const errorData = await response.json();
          errorDetail = errorData.message || errorData.error || String(errorData);
        } catch (e) {
          errorDetail = response.statusText;
        }

        // Return a formatted error response
        return NextResponse.json(
          { error: `Backend API error: ${errorDetail}` }, 
          { status: response.status }
        );
      }

      // Return the successful response
      const data = await response.json();
      return NextResponse.json(data);
      
    } catch (error: any) {
      // Clear the timeout if there's an error
      clearTimeout(timeoutId);
      
      console.error('Backend fetch error:', error);
      
      if (error.name === 'AbortError') {
        return NextResponse.json({ 
          error: 'Request timed out after 10 minutes. Video generation is taking longer than expected.'
        }, { status: 504 });
      }
      
      return NextResponse.json({ 
        error: 'Failed to connect to video generation service',
        details: error.message
      }, { status: 502 });
    }
  } catch (error: any) {
    console.error('API route error:', error);
    return NextResponse.json({ 
      error: 'Error processing request',
      message: error.message 
    }, { status: 500 });
  }
}