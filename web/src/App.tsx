import { useState } from 'react'
import Header from './components/Header'
import Footer from './components/Footer'
import ImageComparison from './components/ImageComparison'
import PhotoInput from './components/PhotoInput'
import DifficultyDropdown from './components/DifficultyDropdown'
import { ChevronDown } from 'lucide-react'
import beforeImage from './images/beforeImage.png'
import afterImage from './images/afterImage.png'

function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [difficulty, setDifficulty] = useState('보통')
  const difficulties = ['쉬움', '보통', '어려움']

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
                beforeImage={beforeImage}
                afterImage={afterImage}
              />
            </div>

            <div className="flex flex-col items-center gap-6 md:gap-8">
              <PhotoInput image={uploadedImage} setImage={setUploadedImage} />

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

                <button className="cursor-pointer rounded-lg bg-gray-900 px-10 py-4 text-gray-50 transition-all hover:bg-gray-800 active:scale-95">
                  <span className="text-base font-medium break-keep text-gray-50">
                    변환
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  )
}

export default App
