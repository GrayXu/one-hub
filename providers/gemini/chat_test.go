package gemini

import (
	"one-api/types"
	"testing"
)

func TestConvertFromChatOpenai_ThinkingLevel(t *testing.T) {
	tests := []struct {
		name              string
		effort            string
		expectedLevel     string
		shouldSetLevel    bool
	}{
		{
			name:           "effort low converts to LOW",
			effort:         "low",
			expectedLevel:  "LOW",
			shouldSetLevel: true,
		},
		{
			name:           "effort medium converts to MEDIUM",
			effort:         "medium",
			expectedLevel:  "MEDIUM",
			shouldSetLevel: true,
		},
		{
			name:           "effort high converts to HIGH",
			effort:         "high",
			expectedLevel:  "HIGH",
			shouldSetLevel: true,
		},
		{
			name:           "invalid effort is ignored",
			effort:         "invalid",
			expectedLevel:  "",
			shouldSetLevel: false,
		},
		{
			name:           "empty effort is ignored",
			effort:         "",
			expectedLevel:  "",
			shouldSetLevel: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maxTokens := 5000
			request := &types.ChatCompletionRequest{
				Model: "gemini-2.0-flash-thinking-exp-01-21",
				Messages: []types.ChatCompletionMessage{
					{
						Role:    "user",
						Content: "Test message",
					},
				},
				Reasoning: &types.ChatReasoning{
					MaxTokens: maxTokens,
					Effort:    tt.effort,
				},
			}

			geminiRequest, err := ConvertFromChatOpenai(request)

			if err != nil {
				t.Fatalf("ConvertFromChatOpenai returned error: %v", err)
			}

			if geminiRequest.GenerationConfig.ThinkingConfig == nil {
				t.Fatal("ThinkingConfig should not be nil when Reasoning is provided")
			}

			if geminiRequest.GenerationConfig.ThinkingConfig.ThinkingBudget == nil {
				t.Fatal("ThinkingBudget should not be nil")
			}

			if *geminiRequest.GenerationConfig.ThinkingConfig.ThinkingBudget != maxTokens {
				t.Errorf("ThinkingBudget = %d, want %d",
					*geminiRequest.GenerationConfig.ThinkingConfig.ThinkingBudget, maxTokens)
			}

			if tt.shouldSetLevel {
				if geminiRequest.GenerationConfig.ThinkingConfig.ThinkingLevel != tt.expectedLevel {
					t.Errorf("ThinkingLevel = %s, want %s",
						geminiRequest.GenerationConfig.ThinkingConfig.ThinkingLevel, tt.expectedLevel)
				}
			} else {
				if geminiRequest.GenerationConfig.ThinkingConfig.ThinkingLevel != "" {
					t.Errorf("ThinkingLevel should be empty for invalid effort, got %s",
						geminiRequest.GenerationConfig.ThinkingConfig.ThinkingLevel)
				}
			}
		})
	}
}

func TestConvertFromChatOpenai_NoReasoning(t *testing.T) {
	request := &types.ChatCompletionRequest{
		Model: "gemini-2.0-flash-thinking-exp-01-21",
		Messages: []types.ChatCompletionMessage{
			{
				Role:    "user",
				Content: "Test message",
			},
		},
		Reasoning: nil,
	}

	geminiRequest, err := ConvertFromChatOpenai(request)

	if err != nil {
		t.Fatalf("ConvertFromChatOpenai returned error: %v", err)
	}

	if geminiRequest.GenerationConfig.ThinkingConfig != nil {
		t.Error("ThinkingConfig should be nil when Reasoning is not provided")
	}
}
