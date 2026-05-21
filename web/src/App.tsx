import { useState } from 'react'
import Header from './components/Header'
import ImageComparison from './components/ImageComparison'
import PhotoInput from './components/PhotoInput'

function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)

  return (
    <div className="h-full min-h-screen max-w-full min-w-fit bg-gray-50">
      <Header />
      <div className="container mx-auto pt-20">
        <div className="flex w-full flex-col items-center gap-16 px-8 py-16">
          <div className="flex flex-col items-center gap-8">
            <p className="text-center text-6xl font-bold break-keep text-gray-900">
              사진으로 만드는 나만의 컬러링북
            </p>
            <p className="max-w-[720px] text-center text-xl font-medium text-balance break-keep text-gray-500">
              소중한 사진을 업로드하고 색상 수를 지정해보세요. 누구나 쉽게 따라
              칠할 수 있는 번호 기반의 컬러링 도안을 즉시 만들어 드립니다.
            </p>
          </div>

          <div className="grid w-full grid-cols-1 gap-8 lg:grid-cols-2">
            <ImageComparison
              beforeImage="https://images.unsplash.com/photo-1541963463532-d68292c34b19?q=80&w=1000&auto=format&fit=crop"
              afterImage="https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1000&auto=format&fit=crop"
            />

            <PhotoInput image={uploadedImage} setImage={setUploadedImage} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
