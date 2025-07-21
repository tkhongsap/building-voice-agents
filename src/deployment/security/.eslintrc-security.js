// ESLint Security Configuration for Voice Agents Platform
// Frontend security linting rules

module.exports = {
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'plugin:security/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking'
  ],
  
  plugins: [
    'security',
    '@typescript-eslint',
    'import'
  ],
  
  env: {
    browser: true,
    node: true,
    es2022: true
  },
  
  parser: '@typescript-eslint/parser',
  
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    project: './tsconfig.json',
    ecmaFeatures: {
      jsx: true
    }
  },
  
  rules: {
    // Security rules
    'security/detect-buffer-noassert': 'error',
    'security/detect-child-process': 'error',
    'security/detect-disable-mustache-escape': 'error',
    'security/detect-eval-with-expression': 'error',
    'security/detect-new-buffer': 'error',
    'security/detect-no-csrf-before-method-override': 'error',
    'security/detect-non-literal-fs-filename': 'warn',
    'security/detect-non-literal-regexp': 'warn',
    'security/detect-non-literal-require': 'error',
    'security/detect-object-injection': 'warn',
    'security/detect-possible-timing-attacks': 'warn',
    'security/detect-pseudoRandomBytes': 'error',
    'security/detect-unsafe-regex': 'error',
    
    // Prevent dangerous global variables
    'no-global-assign': 'error',
    'no-implicit-globals': 'error',
    'no-implied-eval': 'error',
    'no-new-func': 'error',
    
    // Prevent prototype pollution
    'no-proto': 'error',
    'no-extend-native': 'error',
    
    // Input validation
    'no-eval': 'error',
    'no-script-url': 'error',
    
    // XSS prevention
    'no-innerHTML': 'off', // Custom rule if needed
    
    // CSRF protection
    'no-unsafe-finally': 'error',
    
    // Information disclosure
    'no-console': 'warn',
    'no-debugger': 'error',
    'no-alert': 'error',
    
    // Code injection prevention
    'import/no-dynamic-require': 'error',
    
    // TypeScript specific security rules
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unsafe-assignment': 'warn',
    '@typescript-eslint/no-unsafe-call': 'warn',
    '@typescript-eslint/no-unsafe-member-access': 'warn',
    '@typescript-eslint/no-unsafe-return': 'warn',
    '@typescript-eslint/prefer-readonly': 'warn',
    '@typescript-eslint/restrict-template-expressions': 'warn'
  },
  
  overrides: [
    {
      // Test files can be less strict
      files: ['**/*.test.{js,ts,tsx}', '**/__tests__/**/*.{js,ts,tsx}'],
      rules: {
        'security/detect-non-literal-fs-filename': 'off',
        'security/detect-child-process': 'off',
        'no-console': 'off'
      }
    },
    {
      // Configuration files
      files: ['**/*.config.{js,ts}', '**/config/**/*.{js,ts}'],
      rules: {
        'security/detect-non-literal-require': 'off'
      }
    }
  ],
  
  settings: {
    'import/resolver': {
      typescript: {
        alwaysTryTypes: true,
        project: './tsconfig.json'
      }
    }
  },
  
  ignorePatterns: [
    'node_modules/',
    'dist/',
    'build/',
    '.next/',
    'coverage/',
    '*.min.js',
    'public/',
    '.eslintrc.js'
  ]
};