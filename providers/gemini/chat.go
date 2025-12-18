package gemini

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"one-api/common"
	"one-api/common/config"
	"one-api/common/requester"
	"one-api/common/utils"
	"one-api/types"
	"strings"

	"github.com/gin-gonic/gin"
)

type GeminiChatRequest struct {
	Contents         []GeminiChatContent         `json:"contents"`
	SafetySettings   []GeminiChatSafetySettings  `json:"safety_settings,omitempty"`
	GenerationConfig GeminiChatGenerationConfig  `json:"generation_config,omitempty"`
	Tools            []GeminiChatTools           `json:"tools,omitempty"`
	SystemInstruction *GeminiChatContent        `json:"system_instruction,omitempty"`
}

type GeminiChatTools struct {
	FunctionDeclarations any `json:"function_declarations,omitempty"`
	CodeExecution        any `json:"code_execution,omitempty"`
}

type GeminiInlineData struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

type GeminiFileData struct {
	MimeType string `json:"mime_type"`
	FileUri  string `json:"file_uri"`
}

type GeminiPart struct {
	Text         string            `json:"text,omitempty"`
	InlineData   *GeminiInlineData `json:"inline_data,omitempty"`
	FileData     *GeminiFileData   `json:"file_data,omitempty"`
	FunctionCall *any              `json:"functionCall,omitempty"`
}

type GeminiChatContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiChatSafetySettings struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

type GeminiChatGenerationConfig struct {
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	TopK             *int            `json:"top_k,omitempty"`
	MaxOutputTokens  *int            `json:"max_output_tokens,omitempty"`
	CandidateCount   int             `json:"candidate_count,omitempty"`
	StopSequences    []string        `json:"stop_sequences,omitempty"`
	ResponseMimeType string          `json:"response_mime_type,omitempty"`
	ResponseSchema   any             `json:"response_schema,omitempty"`
	ThinkingConfig   *ThinkingConfig `json:"thinking_config,omitempty"`
}

type ThinkingConfig struct {
	ThinkingBudget *int   `json:"thinking_budget,omitempty"`
	ThinkingLevel  string `json:"thinking_level,omitempty"`
}

type GeminiChatResponse struct {
	Candidates     []GeminiChatCandidate    `json:"candidates"`
	PromptFeedback *GeminiChatPromptFeedback `json:"promptFeedback"`
	UsageMetadata  *GeminiUsageMetadata     `json:"usageMetadata"`
}

type GeminiChatCandidate struct {
	Content       GeminiChatContent        `json:"content"`
	FinishReason  string                   `json:"finishReason"`
	Index         int64                    `json:"index"`
	SafetyRatings []GeminiChatSafetyRating `json:"safetyRatings"`
}

type GeminiChatSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

type GeminiChatPromptFeedback struct {
	SafetyRatings []GeminiChatSafetyRating `json:"safetyRatings"`
	BlockReason   string                   `json:"blockReason,omitempty"`
}

type GeminiUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type GeminiErrorResponse struct {
	Error GeminiError `json:"error"`
}

type GeminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

func (p *GeminiProvider) CreateChatCompletion(request *types.ChatCompletionRequest, c *gin.Context) (*types.ChatCompletionResponse, *types.OpenAIErrorWithStatusCode) {
	geminiRequest := ConvertFromChatOpenai(request)
	req, errWithCode := p.GetRequestTextBody(config.RelayModeChatCompletions, request.Model, geminiRequest)
	if errWithCode != nil {
		return nil, errWithCode
	}
	defer req.Body.Close()

	geminiResponse := &GeminiChatResponse{}
	_, errWithCode = p.Requester.SendRequest(req, geminiResponse, false)
	if errWithCode != nil {
		return nil, errWithCode
	}

	return ConvertToChatOpenai(geminiResponse, request)
}

func (p *GeminiProvider) CreateChatCompletionStream(request *types.ChatCompletionRequest, c *gin.Context) (requester.StreamReaderInterface[string], *types.OpenAIErrorWithStatusCode) {
	geminiRequest := ConvertFromChatOpenai(request)
	req, errWithCode := p.GetRequestTextBody(config.RelayModeChatCompletions, request.Model, geminiRequest)
	if errWithCode != nil {
		return nil, errWithCode
	}
	defer req.Body.Close()

	// 请求流式响应
	resp, errWithCode := p.Requester.SendRequestRaw(req)
	if errWithCode != nil {
		return nil, errWithCode
	}

	chatHandler := &GeminiStreamHandler{
		Usage:      p.Usage,
		Request:    request,
		StreamType: "chat",
	}

	return requester.RequestStream(p.Requester, resp, chatHandler.HandlerStream)
}

func ConvertFromChatOpenai(request *types.ChatCompletionRequest) *GeminiChatRequest {
	geminiRequest := GeminiChatRequest{
		Contents:         make([]GeminiChatContent, 0, len(request.Messages)),
		GenerationConfig: GeminiChatGenerationConfig{},
	}

	if request.Reasoning != nil {
		thinkingConfig := &ThinkingConfig{}
		
		// Only set ThinkingBudget if MaxTokens > 0
		if request.Reasoning.MaxTokens > 0 {
			thinkingConfig.ThinkingBudget = &request.Reasoning.MaxTokens
		}
		
		// Convert effort to thinkingLevel
		if request.Reasoning.Effort != "" {
			effortToLevelMap := map[string]string{
				"low":    "LOW",
				"medium": "MEDIUM",
				"high":   "HIGH",
			}
			if level, ok := effortToLevelMap[request.Reasoning.Effort]; ok {
				thinkingConfig.ThinkingLevel = level
			}
		}
		
		// Only set ThinkingConfig if at least one parameter is set
		if thinkingConfig.ThinkingBudget != nil || thinkingConfig.ThinkingLevel != "" {
			geminiRequest.GenerationConfig.ThinkingConfig = thinkingConfig
		}
	}

	shouldAddDummyModelMessage := false
	for _, message := range request.Messages {
		content := GeminiChatContent{
			Role:  message.Role,
			Parts: make([]GeminiPart, 0, len(message.Content)),
		}
		if message.Role == types.ChatMessageRoleSystem {
			content.Role = types.ChatMessageRoleUser
			shouldAddDummyModelMessage = true
		}
		if message.Role == types.ChatMessageRoleAssistant {
			content.Role = "model"
		}
		if message.Role == types.ChatMessageRoleFunction {
			content.Role = "function"
		}
		openaiContent := message.ParseContent()
		for _, part := range openaiContent {
			if part.Type == types.ContentTypeText {
				content.Parts = append(content.Parts, GeminiPart{
					Text: part.Text,
				})
			} else if part.Type == types.ContentTypeImageURL {
				mimeType, data, _ := utils.GetImageFromUrl(part.ImageURL.URL)
				content.Parts = append(content.Parts, GeminiPart{
					InlineData: &GeminiInlineData{
						MimeType: mimeType,
						Data:     data,
					},
				})
			}
		}
		geminiRequest.Contents = append(geminiRequest.Contents, content)

		if shouldAddDummyModelMessage {
			geminiRequest.Contents = append(geminiRequest.Contents, GeminiChatContent{
				Role: "model",
				Parts: []GeminiPart{
					{
						Text: "Understood",
					},
				},
			})
			shouldAddDummyModelMessage = false
		}
	}

	if request.Temperature != nil {
		geminiRequest.GenerationConfig.Temperature = request.Temperature
	}
	if request.TopP != nil {
		geminiRequest.GenerationConfig.TopP = request.TopP
	}
	if request.MaxTokens != nil {
		geminiRequest.GenerationConfig.MaxOutputTokens = request.MaxTokens
	}
	if request.Stop != nil {
		geminiRequest.GenerationConfig.StopSequences = request.Stop
	}

	if request.Tools != nil {
		geminiRequest.Tools = convertToolsToGemini(request.Tools)
	}

	if request.ResponseFormat != nil && request.ResponseFormat.Type == types.ResponseFormatTypeJSONObject {
		geminiRequest.GenerationConfig.ResponseMimeType = "application/json"
	}

	return &geminiRequest
}

func convertToolsToGemini(tools []types.ChatCompletionTool) []GeminiChatTools {
	geminiTools := make([]GeminiChatTools, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == types.ChatMessageRoleFunction {
			geminiTools = append(geminiTools, GeminiChatTools{
				FunctionDeclarations: tool.Function,
			})
		}
	}
	return geminiTools
}

func ConvertToChatOpenai(response *GeminiChatResponse, request *types.ChatCompletionRequest) (*types.ChatCompletionResponse, *types.OpenAIErrorWithStatusCode) {
	fullTextResponse := types.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", utils.GetUUID()),
		Object:  "chat.completion",
		Created: utils.GetTimestamp(),
		Choices: make([]types.ChatCompletionChoice, 0, len(response.Candidates)),
		Model:   request.Model,
	}

	for i, candidate := range response.Candidates {
		choice := types.ChatCompletionChoice{
			Index: int(candidate.Index),
			Message: types.ChatCompletionMessage{
				Role:    types.ChatMessageRoleAssistant,
				Content: "",
			},
			FinishReason: types.FinishReasonStop,
		}
		if len(candidate.Content.Parts) > 0 {
			choice.Message.Content = candidate.Content.Parts[0].Text
		}
		choice.FinishReason = ConvertFinishReason(candidate.FinishReason)
		fullTextResponse.Choices = append(fullTextResponse.Choices, choice)

		if i == 0 && response.UsageMetadata != nil {
			fullTextResponse.Usage = types.Usage{
				PromptTokens:     response.UsageMetadata.PromptTokenCount,
				CompletionTokens: response.UsageMetadata.CandidatesTokenCount,
				TotalTokens:      response.UsageMetadata.TotalTokenCount,
			}
		}
	}

	return &fullTextResponse, nil
}

func ConvertFinishReason(reason string) types.FinishReason {
	switch reason {
	case "STOP":
		return types.FinishReasonStop
	case "MAX_TOKENS":
		return types.FinishReasonLength
	case "SAFETY":
		return types.FinishReasonContentFilter
	case "RECITATION":
		return types.FinishReasonContentFilter
	default:
		return types.FinishReasonNull
	}
}

type GeminiStreamHandler struct {
	Usage      *types.Usage
	Request    *types.ChatCompletionRequest
	StreamType string
}

func (h *GeminiStreamHandler) HandlerStream(rawLine *[]byte, dataChan chan string, errChan chan error) {
	if rawLine == nil || len(*rawLine) == 0 {
		return
	}

	var geminiResp GeminiChatResponse
	err := json.Unmarshal(*rawLine, &geminiResp)
	if err != nil {
		errChan <- common.ErrorToOpenAIError(err)
		return
	}

	if len(geminiResp.Candidates) == 0 {
		return
	}

	h.convertToOpenaiStream(&geminiResp, dataChan)
}

func (h *GeminiStreamHandler) convertToOpenaiStream(geminiResp *GeminiChatResponse, dataChan chan string) {
	streamResponse := types.ChatCompletionStreamResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", utils.GetUUID()),
		Object:  "chat.completion.chunk",
		Created: utils.GetTimestamp(),
		Model:   h.Request.Model,
		Choices: make([]types.ChatCompletionStreamChoice, 0, len(geminiResp.Candidates)),
	}

	for _, candidate := range geminiResp.Candidates {
		choice := types.ChatCompletionStreamChoice{
			Index: int(candidate.Index),
			Delta: types.ChatCompletionStreamChoiceDelta{
				Role:    types.ChatMessageRoleAssistant,
				Content: "",
			},
		}
		if len(candidate.Content.Parts) > 0 {
			choice.Delta.Content = candidate.Content.Parts[0].Text
		}
		choice.FinishReason = ConvertFinishReason(candidate.FinishReason)
		streamResponse.Choices = append(streamResponse.Choices, choice)
	}

	if geminiResp.UsageMetadata != nil {
		h.Usage.PromptTokens = geminiResp.UsageMetadata.PromptTokenCount
		h.Usage.CompletionTokens = geminiResp.UsageMetadata.CandidatesTokenCount
		h.Usage.TotalTokens = geminiResp.UsageMetadata.TotalTokenCount
	}

	responseBody, _ := json.Marshal(streamResponse)
	dataChan <- string(responseBody)
}

func (p *GeminiProvider) GetRequestTextBody(relayMode int, modelName string, geminiRequest *GeminiChatRequest) (*http.Request, *types.OpenAIErrorWithStatusCode) {
	jsonData, err := json.Marshal(geminiRequest)
	if err != nil {
		return nil, common.ErrorWrapper(err, "marshal_request_body_failed", http.StatusInternalServerError)
	}

	fullRequestURL := p.GetFullRequestURL(relayMode, modelName)
	req, err := http.NewRequest(http.MethodPost, fullRequestURL, strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, common.ErrorWrapper(err, "new_request_failed", http.StatusInternalServerError)
	}

	req.Header.Set("Content-Type", "application/json")

	return req, nil
}

func (p *GeminiProvider) GetFullRequestURL(relayMode int, modelName string) string {
	baseURL := strings.TrimSuffix(p.Channel.BaseURL, "/")

	var action string
	if relayMode == config.RelayModeChatCompletions {
		action = "generateContent"
		if p.Channel.Config.Stream {
			action = "streamGenerateContent?alt=sse"
		}
	}

	return fmt.Sprintf("%s/v1beta/models/%s:%s", baseURL, modelName, action)
}

func (p *GeminiProvider) CreateEmbeddings(request *types.EmbeddingRequest, c *gin.Context) (*types.EmbeddingResponse, *types.OpenAIErrorWithStatusCode) {
	// Gemini embedding implementation
	return nil, common.ErrorWrapper(fmt.Errorf("embeddings not supported"), "not_implemented", http.StatusNotImplemented)
}

func (p *GeminiProvider) CreateSpeech(request *types.CreateSpeechRequest, c *gin.Context) (io.ReadCloser, *types.OpenAIErrorWithStatusCode) {
	return nil, common.ErrorWrapper(fmt.Errorf("speech not supported"), "not_implemented", http.StatusNotImplemented)
}

func (p *GeminiProvider) CreateTranscription(request *types.AudioRequest, c *gin.Context) (*types.AudioResponse, *types.OpenAIErrorWithStatusCode) {
	return nil, common.ErrorWrapper(fmt.Errorf("transcription not supported"), "not_implemented", http.StatusNotImplemented)
}

func (p *GeminiProvider) CreateTranslation(request *types.AudioRequest, c *gin.Context) (*types.AudioResponse, *types.OpenAIErrorWithStatusCode) {
	return nil, common.ErrorWrapper(fmt.Errorf("translation not supported"), "not_implemented", http.StatusNotImplemented)
}

func (p *GeminiProvider) CreateImageEdits(request *types.ImageEditRequest, c *gin.Context) (*types.ImageResponse, *types.OpenAIErrorWithStatusCode) {
	return nil, common.ErrorWrapper(fmt.Errorf("image edits not supported"), "not_implemented", http.StatusNotImplemented)
}
