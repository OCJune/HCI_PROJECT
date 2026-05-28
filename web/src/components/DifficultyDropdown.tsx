import React, { useState } from 'react'

interface DifficultyDropdownProps extends React.ComponentProps<'div'> {
  difficulties: string[]
  setDifficulty: (difficulty: string) => void
}

const DifficultyDropdown = ({
  difficulties,
  setDifficulty,
  children,
  ...props
}: DifficultyDropdownProps) => {
  const [isOpen, setIsOpen] = useState(false)

  const handleToggle = () => setIsOpen((prev) => !prev)

  return (
    <div className="relative" {...props}>
      {React.Children.map(children, (child) => {
        if (
          React.isValidElement<{ onClick?: React.MouseEventHandler }>(child)
        ) {
          return React.cloneElement(child, {
            onClick: (e: React.MouseEvent) => {
              child.props.onClick?.(e)
              handleToggle()
            },
          })
        }
        return child
      })}
      {isOpen && (
        <div className="absolute top-full left-0 z-30 flex w-full flex-col items-center gap-0.5 rounded-md border border-gray-50 bg-white p-0.5 shadow-2xl">
          <ul
            role="listbox"
            className="flex w-full flex-1 flex-col items-center"
          >
            {difficulties.map((difficulty) => (
              <li key={difficulty} role="option" className="w-full">
                <button
                  type="button"
                  onClick={() => {
                    setDifficulty(difficulty)
                    setIsOpen(false)
                  }}
                  className="flex w-full cursor-pointer items-center justify-start rounded-md px-4 py-[9.5px] text-base font-medium text-gray-700 hover:bg-gray-50"
                >
                  <span className="truncate">{difficulty}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default DifficultyDropdown
