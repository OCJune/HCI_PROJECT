import { useState } from 'react'
import Header from './components/Header'
import Footer from './components/Footer'
import ImageComparison from './components/ImageComparison'
import PhotoInput from './components/PhotoInput'
import DifficultyDropdown from './components/DifficultyDropdown'
import { ChevronDown } from 'lucide-react'
import beforeImage from './images/beforeImage.png'
import afterImage from './images/afterImage.png'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

interface GenerateResponse {
  original_url: string
  coloring_url: string
  palette_url: string
  combined_url: string
  download_url: string
  download_urls: {
    coloring: string
    combined: string
    palette: string
  }
  difficulty: string
  k: number
}

function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [difficulty, setDifficulty] = useState('보통')
  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const difficulties = ['쉬움', '보통', '어려움']

  const assetUrl = (path: string) => `${API_BASE_URL}${path}`

  const handleGenerate = async () => {
    if (!uploadedFile) {
      setErrorMessage('컬러링북으로 만들 사진을 먼저 선택해주세요.')
      return
    }

    setIsGenerating(true)
    setErrorMessage(null)

    const formData = new FormData()
    formData.append('image', uploadedFile)
    formData.append('difficulty', difficulty)

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json().catch(() => null)
        throw new Error(error?.detail ?? '컬러링북 생성에 실패했습니다.')
      }

      const data = (await response.json()) as GenerateResponse
      setResult(data)
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : '컬러링북 생성에 실패했습니다.',
      )
    } finally {
      setIsGenerating(false)
    }
  }

  const comparisonBefore = result
    ? assetUrl(result.original_url)
    : uploadedImage ?? beforeImage
  const comparisonAfter = result ? assetUrl(result.coloring_url) : afterImage

  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      <Header />
      <main className="container mx-auto flex-grow px-4 pt-24 md:px-8 md:pt-32 lg:pt-40">
        <div className="flex w-full flex-col items-center gap-12 py-12 md:gap-20 md:py-20">
          <div className="flex flex-col items-center gap-6 md:gap-8">
            <p className="text-center text-4xl font-bold break-keep text-gray-900 md:text-5xl lg:text-6xl">
              사진으로 만드는 나만의 컬러링북
            </p>
            <p className="max-w-[720px] text-center text-base font-medium text-balance break-keep text-gray-500 md:text-lg lg:text-xl">
              소중한 사진을 업로드하고 색상 수를 지정해보세요. 누구나 쉽게 따라
              칠할 수 있는 번호 기반의 컬러링 도안을 즉시 만들어 드립니다.
            </p>
          </div>

          <div className="grid w-full grid-cols-1 items-start gap-12 lg:grid-cols-2 lg:gap-8">
            <div className="flex flex-col items-center">
              <ImageComparison
                beforeImage={comparisonBefore}
                afterImage={comparisonAfter}
              />
              {result && (
                <div className="mt-4 flex flex-wrap items-center justify-center gap-4">
                  <a
                    href={assetUrl(result.download_url)}
                    className="rounded-lg bg-blue-600 px-5 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-700"
                  >
                    도안 저장하기
                  </a>
                  <a
                    href={assetUrl(result.combined_url)}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sm font-medium text-blue-600 hover:text-blue-700"
                  >
                    색상표 포함 결과 보기
                  </a>
                </div>
              )}
            </div>

            <div className="flex flex-col items-center gap-6 md:gap-8">
              <PhotoInput
                image={uploadedImage}
                setImage={(image) => {
                  setUploadedImage(image)
                  setResult(null)
                  setErrorMessage(null)
                }}
                setFile={setUploadedFile}
              />

              <div className="flex flex-wrap items-center justify-center gap-4">
                <DifficultyDropdown
                  difficulties={difficulties}
                  setDifficulty={setDifficulty}
                >
                  <button
                    type="button"
                    className="flex min-w-[120px] cursor-pointer items-center gap-2 rounded-lg border border-gray-200 bg-white px-6 py-4 text-gray-700 transition-colors hover:bg-gray-50"
                  >
                    <span className="flex-1 text-base font-medium break-keep">
                      {difficulty}
                    </span>
                    <ChevronDown size={20} className="text-gray-400" />
                  </button>
                </DifficultyDropdown>

                <button
                  type="button"
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="cursor-pointer rounded-lg bg-gray-900 px-10 py-4 text-gray-50 transition-all hover:bg-gray-800 active:scale-95 disabled:cursor-not-allowed disabled:bg-gray-400 disabled:active:scale-100"
                >
                  <span className="text-base font-medium break-keep text-gray-50">
                    {isGenerating ? '변환 중' : '변환'}
                  </span>
                </button>
              </div>

              {errorMessage && (
                <p className="max-w-md text-center text-sm font-medium break-keep text-red-500">
                  {errorMessage}
                </p>
              )}

              {result && (
                <p className="text-sm font-medium text-gray-500">
                  {result.difficulty} 난이도, {result.k}색으로 생성되었습니다.
                </p>
              )}
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  )
}

export default App
